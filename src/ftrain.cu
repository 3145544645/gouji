
// ============================================================================
//  CANTICLE full-model training.
//
//  Forward here is a TRAINING forward: it keeps fp32 activations so the
//  backward pass has what it needs. The inference forward in model.cpp stays
//  the lean fp16 version -- training speed is not what this project is
//  optimising, decode speed is.
//
//  Everything that multiplies a weight matrix multiplies a TERNARY one:
//    fwd:  y = scale * (P - M) @ x
//    bwd:  dx = (P - M)^T @ (dy * scale)     [weights are frozen, STE]
//          dW = (dy * scale) outer x         [accumulated into fp32 master]
// ============================================================================
#include "canticle/ftrain.h"
#include "canticle/ops.h"
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

namespace canticle {

// ---------------------------------------------------------------------------
//  fp32 ternary GEMM forward:  Y[M x N] = X[M x K] @ W^T  where W is [N x K]
//  One block per (row, tile of N). K is strided by threads.
// ---------------------------------------------------------------------------
__global__ void k_tgemm_fwd(const float* __restrict__ X,
                            const uint32_t* __restrict__ P,
                            const uint32_t* __restrict__ M,
                            const float* __restrict__ scale,   // [N]
                            float* __restrict__ Y,
                            int M_, int N, int K) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const int row = blockIdx.y;
  const int WPR = K / 32;
  const uint32_t* pr = P + (size_t)n * WPR;
  const uint32_t* mr = M + (size_t)n * WPR;
  const float* xr = X + (size_t)row * K;
  float acc = 0.f;
  for (int g = 0; g < WPR; ++g) {
    const uint32_t pw = pr[g], mw = mr[g];
    const float* xg = xr + g * 32;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
      uint32_t bit = 1u << j;
      float w = (float)((pw & bit) != 0) - (float)((mw & bit) != 0);
      acc += w * xg[j];
    }
  }
  Y[(size_t)row * N + n] = acc * scale[n];
}

void launch_tgemm_fwd(const float* X, const uint32_t* P, const uint32_t* M,
                      const float* scale, float* Y, int M_, int N, int K, cudaStream_t s) {
  dim3 blk(128);
  dim3 grd((N + 127) / 128, M_);
  k_tgemm_fwd<<<grd, blk, 0, s>>>(X, P, M, scale, Y, M_, N, K);
}

// ---------------------------------------------------------------------------
//  Backward:  dX[M x K] += (dY[M x N] * scale) @ W   [W is [N x K]]
//  and        dW[N x K] += (dY[:,n] * scale[n]) outer X[:,k]
// ---------------------------------------------------------------------------
__global__ void k_tgemm_dx(const float* __restrict__ dY,
                           const uint32_t* __restrict__ P,
                           const uint32_t* __restrict__ M,
                           const float* __restrict__ scale,
                           float* __restrict__ dX,
                           int M_, int N, int K) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;
  const int row = blockIdx.y;
  const int WPR = K / 32;
  const uint32_t bit = 1u << (k & 31);
  const int g = k >> 5;
  const float* dy = dY + (size_t)row * N;
  float acc = 0.f;
  for (int n = 0; n < N; ++n) {
    float gv = dy[n] * scale[n];
    if (gv == 0.f) continue;
    const uint32_t pw = P[(size_t)n * WPR + g];
    const uint32_t mw = M[(size_t)n * WPR + g];
    float w = (float)((pw & bit) != 0) - (float)((mw & bit) != 0);
    acc += w * gv;
  }
  // callers must zero dX (or dX must be a freshly wiped grad buffer) because
  // each (row,k) is owned by exactly one thread here.
  dX[(size_t)row * K + k] = acc;
}

__global__ void k_tgemm_dW(const float* __restrict__ dY,
                           const float* __restrict__ X,
                           const float* __restrict__ scale,
                           float* __restrict__ dW,     // [N x K]
                           int M_, int N, int K) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * K) return;
  const int n = i / K, k = i % K;
  float acc = 0.f;
  for (int m2 = 0; m2 < M_; ++m2)
    acc += dY[(size_t)m2 * N + n] * X[(size_t)m2 * K + k];
  dW[i] += acc * scale[n];
}

void launch_tgemm_dx(const float* dY, const uint32_t* P, const uint32_t* M,
                     const float* scale, float* dX, int M_, int N, int K, cudaStream_t s) {
  // dX is ASSIGNED element-wise (one thread owns each (row,k)), so it must not
  // be pre-accumulated; callers pass a scratch buffer.
  dim3 blk(128), grd((K + 127) / 128, M_);
  k_tgemm_dx<<<grd, blk, 0, s>>>(dY, P, M, scale, dX, M_, N, K);
}

void launch_tgemm_dW(const float* dY, const float* X, const float* scale,
                     float* dW, int M_, int N, int K, cudaStream_t s) {
  // dW is ACCUMULATED: every (m) row contributes to each weight.
  k_tgemm_dW<<<(unsigned)((N*K + 255) / 256), 256, 0, s>>>(dY, X, scale, dW, M_, N, K);
}

// ---------------------------------------------------------------------------
//  Element-wise gradient clipping. A full global-norm clip would need a
//  reduction across every tensor; element-wise clamping at a sane bound stops
//  the blow-up just as effectively and costs one pass with no sync.
// ---------------------------------------------------------------------------
__global__ void k_nan_count(const float* __restrict__ g, long n, int* __restrict__ out) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = g[i];
  if (v != v || v > 1e30f || v < -1e30f) atomicAdd(out, 1);
}
void launch_nan_count(const float* g, long n, int* out, cudaStream_t s) {
  k_nan_count<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(g, n, out);
}

__global__ void k_clip_grad(float* __restrict__ g, long n, float lim) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = g[i];
  if (v >  lim) g[i] =  lim;
  if (v < -lim) g[i] = -lim;
  if (v != v)   g[i] = 0.f;      // scrub NaN so one bad element cannot spread
}
void launch_clip_grad(float* g, long n, float lim, cudaStream_t s) {
  k_clip_grad<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(g, n, lim);
}

__global__ void k_zero(float* __restrict__ p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 0.f;
}
void launch_zero(float* p, int n, cudaStream_t s) { k_zero<<<(n + 255) / 256, 256, 0, s>>>(p, n); }

// ---------------------------------------------------------------------------
//  Cross entropy over a [L x V] logit slab, one row at a time.
//  Writes the loss and dy = (softmax - onehot), both fp32.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  Softmax cross entropy over one row of [L x V] logits.
//
//  BUG HISTORY: the first version reduced with __shfl_xor_sync only, which
//  reduces WITHIN a warp. With 256 threads that is 8 warps, so the softmax max
//  and denominator each covered only 1/8 of the vocabulary. As soon as the
//  logit spread grew, exp(lg - local_max) overflowed to inf, loss became
//  inf - inf = NaN, and training died around step 3 for no visible reason.
//  Both reductions now go through shared memory across every warp.
// ---------------------------------------------------------------------------
__global__ void k_ce_rows(const float* __restrict__ logits,   // [L x V]
                          const int* __restrict__ target,     // [L]
                          float* __restrict__ dy,             // [L x V]
                          float* __restrict__ losses,         // [L]
                          int L, int V) {
  const int row = blockIdx.x;
  if (row >= L) return;
  const float* lg = logits + (size_t)row * V;
  __shared__ float red[32];

  // ---- pass 1: global max over the whole vocabulary ----
  float mx = -1e30f;
  for (int v = threadIdx.x; v < V; v += blockDim.x) mx = fmaxf(mx, lg[v]);
  #pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x < 32) {
    int nw = (blockDim.x + 31) >> 5;
    float v = (threadIdx.x < nw) ? red[threadIdx.x] : -1e30f;
    #pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  mx = red[0];
  __syncthreads();

  // ---- pass 2: global sum of exp ----
  float sum = 0.f;
  for (int v = threadIdx.x; v < V; v += blockDim.x) sum += __expf(lg[v] - mx);
  #pragma unroll
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    int nw = (blockDim.x + 31) >> 5;
    float v = (threadIdx.x < nw) ? red[threadIdx.x] : 0.f;
    #pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  sum = red[0];

  const int tgt = target[row];
  // tgt < 0 marks a masked (padding) position: no loss, no gradient, and it
  // must not contribute to the batch mean either -- hence loss = 0 rather than
  // an average over the real tokens only. The caller divides by the count of
  // unmasked rows.
  if (tgt < 0) {
    if (threadIdx.x == 0) losses[row] = 0.f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) dy[(size_t)row * V + v] = 0.f;
    return;
  }
  if (threadIdx.x == 0)
    losses[row] = (mx + __logf(sum)) - lg[tgt];
  const float inv = (sum > 0.f) ? (1.0f / sum) : 0.f;
  for (int v = threadIdx.x; v < V; v += blockDim.x) {
    float p = __expf(lg[v] - mx) * inv;
    dy[(size_t)row * V + v] = p - ((v == tgt) ? 1.f : 0.f);
  }
}

void launch_ce_rows(const float* logits, const int* target, float* dy, float* losses,
                    int L, int V, cudaStream_t s) {
  k_ce_rows<<<L, 256, 0, s>>>(logits, target, dy, losses, L, V);
}

// ---------------------------------------------------------------------------
//  AdamW over a flat fp32 tensor (used for embeddings, rms gains, master W).
// ---------------------------------------------------------------------------
__global__ void k_adamw2(float* __restrict__ W, const float* __restrict__ G,
                         float* __restrict__ m, float* __restrict__ v, long n,
                         float lr, float b1, float b2, float eps, float wd,
                         float bc1, float bc2) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = G[i];
  m[i] = b1 * m[i] + (1.f - b1) * g;
  v[i] = b2 * v[i] + (1.f - b2) * g * g;
  float mh = m[i] / bc1, vh = v[i] / bc2;
  W[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * W[i]);
}
void launch_adamw2(float* W, const float* G, float* m, float* v, long n,
                   float lr, float b1, float b2, float eps, float wd,
                   float bc1, float bc2, cudaStream_t s) {
  long blocks = (n + 255) / 256;
  k_adamw2<<<(unsigned)blocks, 256, 0, s>>>(W, G, m, v, n, lr, b1, b2, eps, wd, bc1, bc2);
}

// ---------------------------------------------------------------------------
//  Batch mean of a [L] loss vector -> scalar in element 0 of out.
// ---------------------------------------------------------------------------
// Sum a[0..nsum) and divide by ndiv. nsum < L (only the unmasked rows are
// summed) while ndiv is the count of those rows, so the two are separate.
__global__ void k_mean_reduce(const float* __restrict__ a, float* __restrict__ out, int n) {
  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += a[i];
  #pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  __shared__ float red[32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f; int nw = (blockDim.x + 31) >> 5;
    for (int i = 0; i < nw; ++i) t += red[i];
    out[0] = t / (float)n;
  }
}
void launch_mean_reduce(const float* a, float* out, int n, cudaStream_t s) {
  k_mean_reduce<<<1, 256, 0, s>>>(a, out, n);
}

// ---------------------------------------------------------------------------
//  Copy a flat fp32 tensor into an fp16 tensor (for the embedding lookup table
//  used by the inference forward).
// ---------------------------------------------------------------------------
void launch_copy_f32_to_f16(const float* src, __half* dst, long n, cudaStream_t s) {
  launch_f32_to_f16(src, dst, (int)n, s);
}

// ---------------------------------------------------------------------------
//  Backward through the embedding gather, fp32 version:
//    demb[tok] += dh
// ---------------------------------------------------------------------------
__global__ void k_embed_bwd32(float* __restrict__ demb, const int* __restrict__ ids,
                              const float* __restrict__ dh, int T, int d) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= d) return;
  for (int t = 0; t < T; ++t) demb[(size_t)ids[t] * d + j] += dh[t * d + j];
}
void launch_embed_bwd32(float* demb, const int* ids, const float* dh, int T, int d, cudaStream_t s) {
  k_embed_bwd32<<<(d + 255) / 256, 256, 0, s>>>(demb, ids, dh, T, d);
}

// ---------------------------------------------------------------------------
//  Dense fp32 GEMM for the vocabulary head:  Y[M x N] = X[M x K] @ W[N x K]^T
//  The head stays dense (it is only V*d parameters) so the inference path in
//  model.cpp keeps its fp16 table and needs no change.
// ---------------------------------------------------------------------------
__global__ void k_dgemm_fwd(const float* __restrict__ X, const float* __restrict__ W,
                            float* __restrict__ Y, int M_, int N, int K) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const int m = blockIdx.y;
  const float* xr = X + (size_t)m * K;
  const float* wr = W + (size_t)n * K;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += xr[k] * wr[k];
  Y[(size_t)m * N + n] = acc;
}
void launch_dgemm_fwd(const float* X, const float* W, float* Y, int M_, int N, int K, cudaStream_t s) {
  dim3 blk(128), grd((N + 127) / 128, M_);
  k_dgemm_fwd<<<grd, blk, 0, s>>>(X, W, Y, M_, N, K);
}

__global__ void k_dgemm_dx(const float* __restrict__ dY, const float* __restrict__ W,
                           float* __restrict__ dX, int M_, int N, int K) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;
  const int m = blockIdx.y;
  const float* dy = dY + (size_t)m * N;
  float acc = 0.f;
  for (int n = 0; n < N; ++n) acc += dy[n] * W[(size_t)n * K + k];
  dX[(size_t)m * K + k] = acc;
}
void launch_dgemm_dx(const float* dY, const float* W, float* dX, int M_, int N, int K, cudaStream_t s) {
  dim3 blk(128), grd((K + 127) / 128, M_);
  k_dgemm_dx<<<grd, blk, 0, s>>>(dY, W, dX, M_, N, K);
}

__global__ void k_dgemm_dW(const float* __restrict__ dY, const float* __restrict__ X,
                           float* __restrict__ dW, int M_, int N, int K) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * K) return;
  const int n = i / K, k = i % K;
  float acc = 0.f;
  for (int m = 0; m < M_; ++m) acc += dY[(size_t)m * N + n] * X[(size_t)m * K + k];
  dW[i] += acc;
}
void launch_dgemm_dW(const float* dY, const float* X, float* dW, int M_, int N, int K, cudaStream_t s) {
  k_dgemm_dW<<<(unsigned)((N*K + 255) / 256), 256, 0, s>>>(dY, X, dW, M_, N, K);
}

} // namespace canticle

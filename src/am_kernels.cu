
// ============================================================================
//  CANTICLE-AM GPU retrieval kernels.
//  The memory is a narrow, L2-resident matrix; retrieval is one dot-product
//  pass, so the whole read costs a single GEMV -- the same operation already
//  measured at 98% of DRAM peak by the ternary kernels in this repo.
// ============================================================================
#include "canticle/am.h"
#include "canticle/kernels.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace canticle {

// ---------------------------------------------------------------------------
//  Similarity search: one WARP per stored key, lanes split the dimensions.
//
//  WHY: the first version gave each thread a whole row. Thread t reads
//  K[t][0..D), so consecutive threads are D*4 bytes apart -- every load is a
//  separate 32-byte sector and the achieved bandwidth was 85 GB/s against a
//  measured 228 GB/s peak (37%). That is the exact failure mode the int4-naive
//  kernel in this repo had, and it has the same fix: make the warp walk the row
//  together so each lane's load is adjacent to its neighbour's.
//
//  Keys are stored fp16 with a small per-entry scale, halving the bytes moved.
//  Retrieval is a similarity ranking, so ~3 decimal digits of precision is far
//  more than the decision needs.
// ---------------------------------------------------------------------------
__global__ void k_dot_warp(const __half* __restrict__ K,
                           const float* __restrict__ q,
                           float* __restrict__ sim, int N, int D) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= N) return;
  const __half* row = K + (size_t)warp * D;
  float acc = 0.f;
  for (int i = lane; i < D; i += 32) acc += __half2float(row[i]) * q[i];
  #pragma unroll
  for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) sim[warp] = acc;
}

__global__ void k_sm(float* __restrict__ s, int N, float beta) {
  __shared__ float red[32];
  float mx = -1e30f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) mx = fmaxf(mx, s[i]);
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
  __syncthreads(); mx = red[0]; __syncthreads();
  float sum = 0.f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) { s[i] = __expf(beta * (s[i] - mx)); sum += s[i]; }
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
  const float inv = 1.0f / (red[0] + 1e-12f);
  for (int i = threadIdx.x; i < N; i += blockDim.x) s[i] *= inv;
}

__global__ void k_back(const __half* __restrict__ K, const float* __restrict__ w,
                       float* __restrict__ qout, int N, int D) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= D) return;
  float acc = 0.f;
  for (int n = 0; n < N; ++n) acc += w[n] * __half2float(K[(size_t)n * D + i]);
  qout[i] = acc;
}

__global__ void k_l2(float* __restrict__ v, int n) {
  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += v[i] * v[i];
  #pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  __shared__ float red[32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f; int nw = (blockDim.x + 31) >> 5;
    for (int i = 0; i < nw; ++i) t += red[i];
    red[0] = 1.0f / (sqrtf(t) + 1e-8f);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) v[i] *= red[0];
}

void am_dot(const __half* K, const float* q, float* sim, int N, int D, cudaStream_t s) {
  int wpb = 4;                       // warps per block
  int blocks = (N + wpb - 1) / wpb;
  k_dot_warp<<<blocks, wpb * 32, 0, s>>>(K, q, sim, N, D);
}

// ===========================================================================
//  Ternary keys + Hamming-style scoring.
//
//  THE IDEA
//  --------
//  Retrieval only RANKS candidates; it never needs the magnitudes. So each key
//  can be held as two bit-planes, P (positive) and M (negative), exactly like
//  the ternary weights elsewhere in this repo. A key costs D/4 bytes instead of
//  D*2 -- a factor of 8 -- and the comparison becomes integer bit work:
//
//      dot = popc((Pq & Pw) | (Mq & Mw)) - popc((Pq & Mw) | (Mq & Pw))
//
//  which is 4 ANDs, 2 ORs and 2 popcounts per 32 dimensions.
//
//  The retrieved dot is in units of "matching dimensions", so it is scaled by
//  the row's reciprocal norm to stay comparable across rows of differing
//  sparsity -- otherwise a dense key would outrank a sparse one by construction.
// ===========================================================================

// fp32 key -> (P, M, invnorm). Threshold is a fraction of the row's mean |x|,
// the same rule used for the weight ternarisation so the behaviour is uniform
// across the codebase.
__global__ void k_quant_ternary(const float* __restrict__ K, int N, int D,
                                uint32_t* __restrict__ P, uint32_t* __restrict__ M,
                                float* __restrict__ invnorm, float thr_frac) {
  const int row = blockIdx.x;
  if (row >= N) return;
  const float* k = K + (size_t)row * D;
  const int WPR = D / 32;
  uint32_t* pr = P + (size_t)row * WPR;
  uint32_t* mr = M + (size_t)row * WPR;

  for (int g = threadIdx.x; g < WPR; g += blockDim.x) { pr[g] = 0u; mr[g] = 0u; }
  __syncthreads();

  float s = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) s += fabsf(k[i]);
  #pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  __shared__ float red[32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f; int nw = (blockDim.x + 31) >> 5;
    for (int i = 0; i < nw; ++i) t += red[i];
    red[0] = thr_frac * t / (float)D;
  }
  __syncthreads();
  const float thr = red[0];

  int nnz = 0;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = k[i];
    const uint32_t bit = 1u << (i & 31);
    if (v > thr)       { atomicOr(&pr[i >> 5], bit); nnz++; }
    else if (v < -thr) { atomicOr(&mr[i >> 5], bit); nnz++; }
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) nnz += __shfl_down_sync(0xffffffffu, nnz, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = (float)nnz;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f; int nw = (blockDim.x + 31) >> 5;
    for (int i = 0; i < nw; ++i) t += red[i];
    // The score is a raw popcount difference, which grows like sqrt(D), so the
    // reported "similarity" was unbounded -- measured values of 1.414, 1.600
    // and 3.615 were printed for a quantity documented as a cosine. A cosine
    // must lie in [-1, 1] and callers compare it against thresholds.
    //
    // Dividing by the number of NON-ZERO dimensions of the row alone still
    // leaves it unbounded, because the query's own sparsity is not accounted
    // for. The self-similarity of a row with itself is nnz, so normalising by
    // nnz bounds a perfect match at exactly 1.0.
    invnorm[row] = (t > 0.f) ? (1.0f / t) : 0.f;
  }
}

// Query fp32 -> (P, M). Same rule, single row.
__global__ void k_quant_query(const float* __restrict__ q, int D,
                              uint32_t* __restrict__ P, uint32_t* __restrict__ M,
                              float thr_frac) {
  const int WPR = D / 32;
  for (int g = threadIdx.x; g < WPR; g += blockDim.x) { P[g] = 0u; M[g] = 0u; }
  __syncthreads();
  float s = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) s += fabsf(q[i]);
  #pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  __shared__ float red[32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f; int nw = (blockDim.x + 31) >> 5;
    for (int i = 0; i < nw; ++i) t += red[i];
    red[0] = thr_frac * t / (float)D;
  }
  __syncthreads();
  const float thr = red[0];
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = q[i];
    const uint32_t bit = 1u << (i & 31);
    if (v > thr)       atomicOr(&P[i >> 5], bit);
    else if (v < -thr) atomicOr(&M[i >> 5], bit);
  }
}

// popcount scoring: one warp per stored key, lanes split the packed words.
__global__ void k_dot_ternary(const uint32_t* __restrict__ KP,
                              const uint32_t* __restrict__ KM,
                              const float* __restrict__ invnorm,
                              const uint32_t* __restrict__ QP,
                              const uint32_t* __restrict__ QM,
                              float* __restrict__ sim, int N, int D, float qnnz) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= N) return;
  const int WPR = D / 32;
  const uint32_t* pr = KP + (size_t)warp * WPR;
  const uint32_t* mr = KM + (size_t)warp * WPR;

  int acc = 0;
  for (int g = lane; g < WPR; g += 32) {
    const uint32_t pw = pr[g], mw = mr[g];
    const uint32_t pq = QP[g], mq = QM[g];
    const uint32_t agree    = (pq & pw) | (mq & mw);
    const uint32_t disagree = (pq & mw) | (mq & pw);
    acc += __popc(agree) - __popc(disagree);
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) {
    // Divide by sqrt(nnz_row * nnz_query): identical ternary vectors then score
    // exactly 1.0 and unrelated ones stay near 0, which is what a cosine means.
    const float qn = qnnz > 0.f ? qnnz : 1.f;
    sim[warp] = (float)acc / sqrtf(invnorm[warp] > 0.f ? (1.0f / invnorm[warp]) * qn : qn);
  }
}

// ---------------------------------------------------------------------------
//  GPU-side gather.
//
//  The first version of the intent-restricted search copied the ENTIRE key
//  matrix to the host, indexed a subset, and copied the subset back -- 52 MB of
//  device-to-host traffic per query. That single line capped the ternary path
//  at 25 GB/s while the arithmetic said it should have been much faster.
//  Gathering on the device turns it into one small kernel reading only the rows
//  that are actually needed.
// ---------------------------------------------------------------------------
__global__ void k_gather_f16(const __half* __restrict__ src, const int* __restrict__ rows,
                             __half* __restrict__ dst, int m, int D) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m * D) return;
  const int r = i / D, c = i % D;
  dst[i] = src[(size_t)rows[r] * D + c];
}

__global__ void k_gather_tern(const uint32_t* __restrict__ P,
                              const uint32_t* __restrict__ M,
                              const float* __restrict__ inv,
                              const int* __restrict__ rows,
                              uint32_t* __restrict__ dP, uint32_t* __restrict__ dM,
                              float* __restrict__ dInv, int m, int WPR) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m * WPR) return;
  const int r = i / WPR, c = i % WPR;
  dP[i] = P[(size_t)rows[r] * WPR + c];
  dM[i] = M[(size_t)rows[r] * WPR + c];
  if (c == 0) dInv[r] = inv[rows[r]];
}

void am_gather_f16(const __half* src, const int* rows, __half* dst, int m, int D,
                   cudaStream_t s) {
  const int n = m * D;
  k_gather_f16<<<(n + 255) / 256, 256, 0, s>>>(src, rows, dst, m, D);
}
void am_gather_tern(const uint32_t* P, const uint32_t* M, const float* inv,
                    const int* rows, uint32_t* dP, uint32_t* dM, float* dInv,
                    int m, int WPR, cudaStream_t s) {
  const int n = m * WPR;
  k_gather_tern<<<(n + 255) / 256, 256, 0, s>>>(P, M, inv, rows, dP, dM, dInv, m, WPR);
}

void am_quantize_keys(const float* K, int N, int D, uint32_t* P, uint32_t* M,
                      float* invnorm, float thr_frac, cudaStream_t s) {
  k_quant_ternary<<<N, 128, 0, s>>>(K, N, D, P, M, invnorm, thr_frac);
}
// nnz of the query is needed by the scorer to form a proper cosine.
__global__ void k_query_nnz(const uint32_t* __restrict__ P, const uint32_t* __restrict__ M,
                            int WPR, float* __restrict__ out) {
  __shared__ int red[64];
  int c = 0;
  for (int g = threadIdx.x; g < WPR; g += blockDim.x)
    c += __popc(P[g]) + __popc(M[g]);
  red[threadIdx.x] = c;
  __syncthreads();
  if (threadIdx.x == 0) {
    int t = 0;
    for (int i = 0; i < (int)blockDim.x; ++i) t += red[i];
    out[0] = (float)t;
  }
}

void am_quantize_query(const float* q, int D, uint32_t* P, uint32_t* M,
                       float thr_frac, float* nnz_out, cudaStream_t s) {
  k_quant_query<<<1, 256, 0, s>>>(q, D, P, M, thr_frac);
  if (nnz_out) k_query_nnz<<<1, 256, 0, s>>>(P, M, D / 32, nnz_out);
}
void am_dot_ternary(const uint32_t* KP, const uint32_t* KM, const float* invnorm,
                    const uint32_t* QP, const uint32_t* QM,
                    float* sim, int N, int D, float qnnz, cudaStream_t s) {
  const int wpb = 4;
  k_dot_ternary<<<(N + wpb - 1) / wpb, wpb * 32, 0, s>>>(KP, KM, invnorm, QP, QM,
                                                          sim, N, D, qnnz);
}

__global__ void k_f32_to_f16(const float* __restrict__ src, __half* __restrict__ dst, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}
void am_to_f16(const float* src, __half* dst, long n, cudaStream_t s) {
  k_f32_to_f16<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(src, dst, n);
}
void am_softmax(float* s, int N, float beta, cudaStream_t st) {
  k_sm<<<1, 256, 0, st>>>(s, N, beta);
}
void am_back(const __half* K, const float* w, float* qout, int N, int D, cudaStream_t s) {
  k_back<<<(D + 255)/256, 256, 0, s>>>(K, w, qout, N, D);
}
void am_l2(float* v, int n, cudaStream_t s) {
  k_l2<<<1, 256, 0, s>>>(v, n);
}

} // namespace canticle

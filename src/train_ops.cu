
// ============================================================================
//  CANTICLE training primitives.
//  Innovation #3 in practice: a full-precision master weight is kept in fp32,
//  ternarised on the fly for the forward pass, and updated with the
//  straight-through gradient. Everything here runs on the device so the host
//  never has to round-trip the weight matrix.
// ============================================================================
#include "canticle/train.h"
#include "canticle/ops.h"
#include <cuda_fp16.h>
#include <cmath>

namespace canticle {

// ---------------------------------------------------------------------------
//  Row-wise ternarisation. One block per output row (K <= 1024 assumed).
//  Rule:  thr = 0.7 * mean(|w|) ;  +1 if w > thr, -1 if w < -thr, else 0.
//  The row scale is the mean |w| of the surviving weights, written as fp32.
// ---------------------------------------------------------------------------
__global__ void k_ternarize_rows(const float* __restrict__ W,
                                 uint32_t* __restrict__ P,
                                 uint32_t* __restrict__ M,
                                 float* __restrict__ scale,
                                 int N, int K) {
  const int row = blockIdx.x;
  if (row >= N) return;
  const float* w = W + (size_t)row * K;

  // pass 1: mean |w|
  float s = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) s += fabsf(w[k]);
  #pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  __shared__ float red[32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    int nw = (blockDim.x + 31) >> 5;
    float v = (threadIdx.x < nw) ? red[threadIdx.x] : 0.f;
    #pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0) red[0] = v / (float)K;
  }
  __syncthreads();
  const float thr = 0.7f * red[0];

  // pass 2: bit planes + magnitude accumulation
  const int WPR = K / 32;
  uint32_t* pr = P + (size_t)row * WPR;
  uint32_t* mr = M + (size_t)row * WPR;
  for (int g = threadIdx.x; g < WPR; g += blockDim.x) { pr[g] = 0u; mr[g] = 0u; }
  __syncthreads();

  float mag = 0.f; int cnt = 0;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = w[k];
    uint32_t bit = 1u << (k & 31);
    uint32_t* dst = nullptr;
    if (v > thr)       { dst = &pr[k >> 5]; mag += fabsf(v); cnt++; }
    else if (v < -thr) { dst = &mr[k >> 5]; mag += fabsf(v); cnt++; }
    if (dst) atomicOr(dst, bit);
  }
  // reduce mag/cnt
  #pragma unroll
  for (int o = 16; o; o >>= 1) { mag += __shfl_down_sync(0xffffffffu, mag, o); cnt += __shfl_down_sync(0xffffffffu, cnt, o); }
  __shared__ float redm[32]; __shared__ int redc[32];
  if ((threadIdx.x & 31) == 0) { redm[threadIdx.x >> 5] = mag; redc[threadIdx.x >> 5] = cnt; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float tm = 0.f; int tc = 0;
    int nw = (blockDim.x + 31) >> 5;
    for (int i = 0; i < nw; ++i) { tm += redm[i]; tc += redc[i]; }
    scale[row] = tc ? tm / (float)tc : 1e-3f;
  }
}

void launch_ternarize_rows(const float* W, uint32_t* P, uint32_t* M, float* scale,
                           int N, int K, cudaStream_t s) {
  k_ternarize_rows<<<N, 128, 0, s>>>(W, P, M, scale, N, K);
}

// ---------------------------------------------------------------------------
//  Outer product for the head gradient:  dW[v][j] = g[v] * h[j]
//  This is the whole backward pass through a linear layer.
// ---------------------------------------------------------------------------
__global__ void k_outer(float* __restrict__ dW, const float* __restrict__ g,
                        const float* __restrict__ h, int N, int K) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * K;
  if (i >= total) return;
  int v = i / K, j = i % K;
  dW[i] = g[v] * h[j];
}

void launch_outer(float* dW, const float* g, const float* h, int N, int K, cudaStream_t s) {
  int total = N * K;
  k_outer<<<(total + 255) / 256, 256, 0, s>>>(dW, g, h, N, K);
}

// ---------------------------------------------------------------------------
//  AdamW on the fp32 master weights.
// ---------------------------------------------------------------------------
__global__ void k_adam(float* __restrict__ W, const float* __restrict__ G,
                       float* __restrict__ m, float* __restrict__ v, int n,
                       float lr, float b1, float b2, float eps, float wd,
                       float bc1, float bc2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = G[i];
  m[i] = b1 * m[i] + (1.f - b1) * g;
  v[i] = b2 * v[i] + (1.f - b2) * g * g;
  float mh = m[i] / bc1, vh = v[i] / bc2;
  W[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * W[i]);
}

void launch_adam(float* W, const float* G, float* m, float* v, int n,
                 float lr, float b1, float b2, float eps, float wd,
                 float bc1, float bc2, cudaStream_t s) {
  k_adam<<<(n + 255) / 256, 256, 0, s>>>(W, G, m, v, n, lr, b1, b2, eps, wd, bc1, bc2);
}

// ---------------------------------------------------------------------------
//  Gather rows of the fp32 master embedding into an fp16 table (vocab x d).
//  Done as a plain copy so the forward pass can use the fp16 tensor cores path.
// ---------------------------------------------------------------------------
__global__ void k_f32_to_f16(const float* __restrict__ src, __half* __restrict__ dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}
void launch_f32_to_f16(const float* src, __half* dst, int n, cudaStream_t s) {
  k_f32_to_f16<<<(n + 255) / 256, 256, 0, s>>>(src, dst, n);
}

__global__ void k_f16_to_f32(const __half* __restrict__ src, float* __restrict__ dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __half2float(src[i]);
}
void launch_f16_to_f32(const __half* src, float* dst, int n, cudaStream_t s) {
  k_f16_to_f32<<<(n + 255) / 256, 256, 0, s>>>(src, dst, n);
}

// ---------------------------------------------------------------------------
//  dW for a ternary layer, straight-through:  dW_fp32 = scale * g x h
//  (scale folded into g before the outer product)
// ---------------------------------------------------------------------------
__global__ void k_scale_vec(const float* __restrict__ g, const float* __restrict__ scale,
                            float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = g[i] * scale[i];
}
void launch_scale_vec(const float* g, const float* scale, float* out, int n, cudaStream_t s) {
  k_scale_vec<<<(n + 255) / 256, 256, 0, s>>>(g, scale, out, n);
}

} // namespace canticle


// ============================================================================
//  CANTICLE training kernels.
// ============================================================================
#include "canticle/train.h"
#include "canticle/ops.h"
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace canticle {

// ---------------------------------------------------------------------------
// k_tern_gemm_dx: dX[k] = sum_n (p_nk - m_nk) * scale[n] * dY[n]
//
// This is the backward pass through a ternary linear layer. Gradients are not
// quantised (STE), so the multiply is done in fp32 against the *dequantised
// sign pattern*. Layout choice: one warp owns one 32-bit plane word, lane l
// owns bit l, so all 32 lanes read the SAME word -> a single broadcast load
// per plane per output row instead of 32 scattered ones.
// ---------------------------------------------------------------------------
__global__ void k_tern_gemm_dx(const uint32_t* __restrict__ P,
                               const uint32_t* __restrict__ M,
                               const float*    __restrict__ dscale,   // scale[n]
                               const float*    __restrict__ dY,       // [N]
                               float*          __restrict__ dX,       // [K]
                               int N, int K) {
  const int laneword = (blockIdx.x * blockDim.x + threadIdx.x) & 31;   // bit index
  const int wordid   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // which word
  const int WPR      = K / 32;
  if (wordid >= WPR) return;

  const uint32_t mask = 1u << laneword;
  float acc = 0.f;
  for (int n = 0; n < N; ++n) {
    const uint32_t pw = __ldg(&P[(size_t)n * WPR + wordid]);
    const uint32_t mw = __ldg(&M[(size_t)n * WPR + wordid]);
    const float    g  = dscale[n] * dY[n];
    if (g == 0.f) continue;
    float w = (float)((pw & mask) != 0) - (float)((mw & mask) != 0);
    acc += w * g;
  }
  dX[wordid * 32 + laneword] = acc;
}

void launch_tern_gemm_dx(const uint32_t* dP, const uint32_t* dM, const float* dscale,
                         const float* dY, float* dX, int N, int K, cudaStream_t s) {
  int words = K / 32;
  int threads = 128;
  int blocks = (words * 32 + threads - 1) / threads;
  k_tern_gemm_dx<<<blocks, threads, 0, s>>>(dP, dM, dscale, dY, dX, N, K);
}

// ---------------------------------------------------------------------------
// k_scale_sens: sensitivity of the loss to each output row's quantisation
// scale,  dscale[n] = dY[n] * raw[n] / scale[n].
// This is what makes the *ternary* path trainable end to end: the scale is the
// only continuous multiplier left after the weights go discrete.
// ---------------------------------------------------------------------------
__global__ void k_scale_sens(const float* __restrict__ raw,
                             const float* __restrict__ dY,
                             float* __restrict__ dscale, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) dscale[n] = dY[n] * raw[n];
}

void launch_scale_sens(const float* raw, const float* dY, float* dscale, int N, cudaStream_t s) {
  k_scale_sens<<<(N + 255) / 256, 256, 0, s>>>(raw, dY, dscale, N);
}

// ---------------------------------------------------------------------------
// Mean pool forward: h[j] = (1/T) sum_t emb[id_t][j]
// ---------------------------------------------------------------------------
__global__ void k_meanpool(const __half* __restrict__ emb, const int* __restrict__ ids,
                           __half* __restrict__ h, int T, int d) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= d) return;
  float acc = 0.f;
  for (int t = 0; t < T; ++t) acc += __half2float(emb[(size_t)ids[t] * d + j]);
  h[j] = __float2half(acc / (float)T);
}

void launch_meanpool(const __half* emb, const int* ids, __half* h, int T, int d, cudaStream_t s) {
  k_meanpool<<<(d + 255) / 256, 256, 0, s>>>(emb, ids, h, T, d);
}

// ---------------------------------------------------------------------------
// Scatter-add for the embedding gradient:  demb[id_t][j] += dh[j] / T
// Multiple lanes may hit the same row; that is benign at this scale because we
// serialise on the host side when ids repeat. For unique ids atomicAdd is used.
// ---------------------------------------------------------------------------
__global__ void k_embed_bwd(float* __restrict__ demb, const int* __restrict__ ids,
                            const float* __restrict__ dh, int T, int d) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= d) return;
  float g = dh[j] / (float)T;
  for (int t = 0; t < T; ++t) atomicAdd(&demb[(size_t)ids[t] * d + j], g);
}

void launch_embed_bwd(float* demb, const int* ids, const float* dh, int T, int d, cudaStream_t s) {
  k_embed_bwd<<<(d + 255) / 256, 256, 0, s>>>(demb, ids, dh, T, d);
}

// ---------------------------------------------------------------------------
// Ternary GEMV with raw-output capture:  y[v] = sum_k w_vk h_k
// We also write raw[v] = y[v] / scale[v] so the backward can recover the
// sensitivity of the scale. Used for the vocabulary head.
// ---------------------------------------------------------------------------
__global__ void k_tern_head(const uint32_t* __restrict__ P,
                            const uint32_t* __restrict__ M,
                            const __half*   __restrict__ h,
                            float*          __restrict__ y,
                            float*          __restrict__ raw,
                            int V, int d) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= V) return;
  const int WPR = d / 32;
  const uint32_t* pr = P + (size_t)v * WPR;
  const uint32_t* mr = M + (size_t)v * WPR;
  float acc = 0.f;
  for (int g = 0; g < WPR; ++g) {
    const uint32_t pw = pr[g], mw = mr[g];
    const __half* hg = h + g * 32;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
      uint32_t bit = 1u << j;
      float w = (float)((pw & bit) != 0) - (float)((mw & bit) != 0);
      acc += w * __half2float(hg[j]);
    }
  }
  raw[v] = acc;
  y[v] = acc;   // scale applied on the host side during CE
}

void launch_tern_head(const uint32_t* dP, const uint32_t* dM, const __half* h,
                      float* y, float* raw, int V, int d, cudaStream_t s) {
  k_tern_head<<<(V + 127) / 128, 128, 0, s>>>(dP, dM, h, y, raw, V, d);
}

} // namespace canticle

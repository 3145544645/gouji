
// ============================================================================
//  CANTICLE model operators: embedding, rmsnorm, rope, SwiGLU, attention,
//  and the ternary GEMV entry points used by the forward pass.
// ============================================================================
#include "canticle/model.h"
#include "canticle/ops.h"
#include <cuda_fp16.h>

namespace canticle {

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  #pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
  return v;
}

// ------------------------------------------------------------- embed lookup
__global__ void k_embed(const __half* __restrict__ tab, int d,
                        const int* __restrict__ tok, __half* __restrict__ out, int n) {
  int t = tok[0];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d; i += gridDim.x * blockDim.x)
    out[i] = tab[(size_t)t * d + i];
}

// ------------------------------------------------------------------ rmsnorm
__global__ void k_rmsnorm(const __half* __restrict__ x, const __half* __restrict__ w,
                          __half* __restrict__ out, int d, float eps) {
  __shared__ float red[32];
  float ss = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float v = __half2float(x[i]); ss += v * v;
  }
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    int nw = (blockDim.x + 31) >> 5;
    float v = (threadIdx.x < nw) ? red[threadIdx.x] : 0.f;
    v = warp_sum(v);
    if (threadIdx.x == 0) red[0] = rsqrtf(v / d + eps);
  }
  __syncthreads();
  float r = red[0];
  for (int i = threadIdx.x; i < d; i += blockDim.x)
    out[i] = __float2half(__half2float(x[i]) * r * __half2float(w[i]));
}

// --------------------------------------------------------------------- rope
__global__ void k_rope(__half* __restrict__ q, __half* __restrict__ k,
                       int n_head, int head_dim, int pos, float theta) {
  int h  = blockIdx.x;
  int i  = threadIdx.x;                 // pair index
  int half = head_dim / 2;
  if (i >= half) return;
  float freq = 1.0f / powf(theta, (2.0f * i) / (float)head_dim);
  float ang  = pos * freq;
  float c = cosf(ang), s = sinf(ang);

  __half* qh = q + (size_t)h * head_dim;
  __half2* q2 = reinterpret_cast<__half2*>(qh);
  float2 qf = __half22float2(q2[i]);
  q2[i] = __floats2half2_rn(qf.x * c - qf.y * s, qf.x * s + qf.y * c);

  __half* kh = k + (size_t)h * head_dim;
  __half2* k2 = reinterpret_cast<__half2*>(kh);
  float2 kf = __half22float2(k2[i]);
  k2[i] = __floats2half2_rn(kf.x * c - kf.y * s, kf.x * s + kf.y * c);
}

// ------------------------------------------------------------------- swiglu
__global__ void k_silu_mul(const __half* __restrict__ g, const __half* __restrict__ u,
                           __half* __restrict__ out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    float x = __half2float(g[i]);
    float s = x / (1.0f + __expf(-x));
    out[i] = __float2half(s * __half2float(u[i]));
  }
}

// ------------------------------------------------------- attention (decode)
// Warp-per-head attention: 32 lanes cover head_dim (<=128) by striding.
__global__ void k_attn_decode_warp(const __half* __restrict__ q,
                                   const __half* __restrict__ kcache,
                                   const __half* __restrict__ vcache,
                                   __half* __restrict__ o,
                                   int T, int n_head, int head_dim, int nkv, float scale) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int NW = blockDim.x >> 5;
  const int h = blockIdx.x * NW + warp;
  if (h >= n_head) return;
  const int kvh = h / (n_head / nkv);      // grouped-query support

  extern __shared__ float sc[];            // [NW][T]
  float* my = sc + (size_t)warp * T;

  const __half* qh = q + (size_t)h * head_dim;
  const __half* kbase = kcache + (size_t)kvh * head_dim;
  const __half* vbase = vcache + (size_t)kvh * head_dim;
  const int kstride = (size_t)nkv * head_dim;

  for (int t = 0; t < T; ++t) {
    float d = 0.f;
    for (int i = lane; i < head_dim; i += 32)
      d += __half2float(qh[i]) * __half2float(kbase[(size_t)t * kstride + i]);
    d = warp_sum(d);
    if (lane == 0) my[t] = d * scale;
  }
  __syncwarp();

  float m = -1e30f;
  for (int t = 0; t < T; ++t) m = fmaxf(m, my[t]);
  float s = 0.f;
  for (int t = 0; t < T; ++t) { float e = __expf(my[t] - m); my[t] = e; s += e; }
  float inv = 1.0f / s;

  __half* oh = o + (size_t)h * head_dim;
  for (int i = lane; i < head_dim; i += 32) {
    float acc = 0.f;
    for (int t = 0; t < T; ++t)
      acc += my[t] * __half2float(vbase[(size_t)t * kstride + i]);
    oh[i] = __float2half(acc * inv);
  }
}

// ------------------------------------------------------- final projection
// logits[v] = dot(lm_head[v], x)  -- fp16 table, bandwidth bound at vocab x d
__global__ void k_lm_head(const __half* __restrict__ head, const __half* __restrict__ x,
                          float* __restrict__ logits, int vocab, int d) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= vocab) return;
  const __half* row = head + (size_t)v * d;
  float acc = 0.f;
  for (int i = 0; i < d; ++i) acc += __half2float(row[i]) * __half2float(x[i]);
  logits[v] = acc;
}

// ================================================================== launchers
void op_embed(const __half* tab, const int* tok, __half* out, int n, int d, cudaStream_t s) {
  k_embed<<<std::min(64, (d + 255) / 256), 256, 0, s>>>(tab, d, tok, out, n);
}
void op_rmsnorm(const __half* x, const __half* w, __half* out, int d, float eps, cudaStream_t s) {
  k_rmsnorm<<<1, 256, 0, s>>>(x, w, out, d, eps);
}
void op_rope(__half* q, __half* k, int n_head, int head_dim, int pos, float theta, cudaStream_t s) {
  k_rope<<<n_head, head_dim / 2, 0, s>>>(q, k, n_head, head_dim, pos, theta);
}
void op_silu_mul(const __half* g, const __half* u, __half* out, int n, cudaStream_t s) {
  k_silu_mul<<<std::min(512, (n + 255) / 256), 256, 0, s>>>(g, u, out, n);
}
void op_attn_decode(const __half* q, const __half* kc, const __half* vc, __half* o,
                    int T, int n_head, int head_dim, int nkv, float scale,
                    int smem, cudaStream_t s) {
  int nw = 4, th = nw * 32;
  int blocks = (n_head + nw - 1) / nw;
  static bool set = false;
  if (!set) { CUDA_CHECK(cudaFuncSetAttribute(k_attn_decode_warp, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); set = true; }
  k_attn_decode_warp<<<blocks, th, smem, s>>>(q, kc, vc, o, T, n_head, head_dim, nkv, scale);
}
void op_lm_head(const __half* head, const __half* x, float* logits, int vocab, int d, cudaStream_t s) {
  k_lm_head<<<(vocab + 255) / 256, 256, 0, s>>>(head, x, logits, vocab, d);
}

// ---- fp16 GEMV for the fp16 paths (embedding-class layers) ----
__global__ void k_gemv_f16v(const __half* __restrict__ W, const __half* __restrict__ x,
                            __half* __restrict__ y, int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const __half* row = W + (size_t)n * K;
  float acc = 0.f;
  for (int i = 0; i < K; ++i) acc += __half2float(row[i]) * __half2float(x[i]);
  y[n] = __float2half(acc);
}
__global__ void k_add_inplace(__half* __restrict__ a, const __half* __restrict__ b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}
void op_add_inplace(__half* a, const __half* b, int n, cudaStream_t s) {
  k_add_inplace<<<(n + 255) / 256, 256, 0, s>>>(a, b, n);
}

void op_gemv_f16(const __half* W, const __half* x, __half* y, int N, int K, cudaStream_t s) {
  k_gemv_f16v<<<(N + 127) / 128, 128, 0, s>>>(W, x, y, N, K);
}

} // namespace canticle


// ============================================================================
//  CANTICLE inference kernels
//    Innovation #1 : bit-plane ternary GEMV (2 bits/weight)
//    Innovation #2 : L2-persistent weight residency
//  Reference points for the A/B: int4 symmetric and plain fp16 HFMA2.
// ============================================================================
#include "canticle/kernels.h"
#include <cuda_fp16.h>

namespace canticle {

__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

template <typename F> static void set_smem(F fn, int bytes) {
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

__device__ __forceinline__ void stage_x(const __half* x, __half* sx, int K) {
  for (int i = threadIdx.x; i < K; i += blockDim.x) sx[i] = x[i];
  __syncthreads();
}

// ------------------------------------------------------- fill / L2 flush
__global__ void fill_kernel(uint32_t* __restrict__ p, long n, uint32_t seed) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long s = (long)gridDim.x * blockDim.x;
  for (; i < n; i += s) {
    uint32_t x = seed + (uint32_t)i * 2654435761u;
    x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    // keep nibbles away from 0x0 to make ternary planes dense
    p[i] = (x & 0x01010101u) | (x & 0x02020202u) | (x * 3u);
  }
}

__global__ void touch_kernel(const uint4* __restrict__ p, long n, uint4* __restrict__ sink) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long s = (long)gridDim.x * blockDim.x;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (; i < n; i += s) { uint4 v = p[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
  if (acc.x == 0xDEADBEEFu) *sink = acc;   // never true; defeats DCE
}

void fill_random(uint32_t* d, size_t words, uint32_t seed, cudaStream_t s) {
  fill_kernel<<<1024, 256, 0, s>>>(d, (long)words, seed);
}
void flush_l2(const void* d, size_t bytes, cudaStream_t s) {
  long n = (long)(bytes / 16);
  touch_kernel<<<2048, 256, 0, s>>>((const uint4*)d, n, (uint4*)nullptr);
}

// ---------------------------------------------------------------- copy probe
__global__ void copy_kernel(const float4* __restrict__ in, float4* __restrict__ out, long n) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long s = (long)gridDim.x * blockDim.x;
  for (; i < n; i += s) out[i] = in[i];
}

void copy_bandwidth_probe(float* out_gbps, size_t bytes, int iters, cudaStream_t st) {
  float4 *a = nullptr, *b = nullptr;
  CUDA_CHECK(cudaMalloc(&a, bytes));
  CUDA_CHECK(cudaMalloc(&b, bytes));
  CUDA_CHECK(cudaMemsetAsync(a, 1, bytes, st));
  long n4 = (long)(bytes / 16);
  int threads = 256, blocks = 1024;
  for (int i = 0; i < 3; i++) copy_kernel<<<blocks, threads, 0, st>>>(a, b, n4);
  CUDA_CHECK(cudaStreamSynchronize(st));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, st);
  for (int i = 0; i < iters; i++) copy_kernel<<<blocks, threads, 0, st>>>(a, b, n4);
  cudaEventRecord(e1, st);
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  *out_gbps = (float)((2.0 * bytes * iters / 1e9) / (ms / 1000.0));
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaFree(a); cudaFree(b);
}

// ============================================================== BASELINE int4
__global__ void k_gemv_int4_naive(const uint32_t* __restrict__ W,
                                  const __half*   __restrict__ x,
                                  float*          __restrict__ y,
                                  int N, int K, float scale) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const uint32_t* row = W + (size_t)n * (K / 8);
  float acc = 0.f;
  for (int g = 0; g < K / 8; ++g) {
    uint32_t w = row[g];
    const __half* xg = x + g * 8;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      int q = (int)((w >> (4 * j)) & 0xF) - 8;
      acc += (float)q * __half2float(xg[j]);
    }
  }
  y[n] = acc * scale;
}

__global__ void k_gemv_int4_warp(const uint32_t* __restrict__ W,
                                 const __half*   __restrict__ x,
                                 float*          __restrict__ y,
                                 int N, int K, float scale) {
  extern __shared__ __half sx[];
  stage_x(x, sx, K);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int NW = blockDim.x >> 5, WPR = K / 8;
  for (int n = blockIdx.x * NW + warp; n < N; n += gridDim.x * NW) {
    const uint32_t* row = W + (size_t)n * WPR;
    float acc = 0.f;
    for (int g = lane; g < WPR; g += 32) {
      uint32_t w = row[g];
      const __half* xg = sx + g * 8;
      #pragma unroll
      for (int j = 0; j < 8; ++j)
        acc += (float)((int)((w >> (4 * j)) & 0xF) - 8) * __half2float(xg[j]);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[n] = acc * scale;
  }
}

// ============ REFERENCE POINT: plain fp16 weights, pure HFMA2 (no unpacking)
// Establishes the *instruction-throughput ceiling*: what a kernel can do when
// every weight is already in the compute-friendly format.
__global__ void k_gemv_f16(const __half* __restrict__ W,
                           const __half* __restrict__ x,
                           float* __restrict__ y, int N, int K) {
  extern __shared__ __half sx[];
  stage_x(x, sx, K);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int NW = blockDim.x >> 5, K2 = K / 2;
  const __half2* x2 = reinterpret_cast<const __half2*>(sx);
  for (int n = blockIdx.x * NW + warp; n < N; n += gridDim.x * NW) {
    const __half2* w2 = reinterpret_cast<const __half2*>(W + (size_t)n * K);
    __half2 acc = __float2half2_rn(0.f);
    for (int g = lane; g < K2; g += 32) acc = __hfma2(w2[g], x2[g], acc);
    float s = warp_reduce_sum(__low2float(acc) + __high2float(acc));
    if (lane == 0) y[n] = s;
  }
}

// ======================= INNOVATION #1: bit-plane ternary, warp-cooperative
// W in {-1,0,+1} held as P,M with w = P - M  ->  2 bits/weight.
__global__ void k_gemv_t3_warp(const uint32_t* __restrict__ P,
                               const uint32_t* __restrict__ M,
                               const __half*   __restrict__ x,
                               float*          __restrict__ y,
                               int N, int K, float scale) {
  extern __shared__ __half sx[];
  stage_x(x, sx, K);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int NW = blockDim.x >> 5, WPR = K / 32;
  for (int n = blockIdx.x * NW + warp; n < N; n += gridDim.x * NW) {
    const uint32_t* pr = P + (size_t)n * WPR;
    const uint32_t* mr = M + (size_t)n * WPR;
    float acc = 0.f;
    for (int g = lane; g < WPR; g += 32) {
      const uint32_t pw = pr[g], mw = mr[g];
      const __half2* xg2 = reinterpret_cast<const __half2*>(sx + g * 32);
      #pragma unroll
      for (int j = 0; j < 16; ++j) {
        uint32_t p2 = (pw >> (2 * j)) & 0x3u;
        uint32_t m2 = (mw >> (2 * j)) & 0x3u;
        float2 f = __half22float2(xg2[j]);
        acc += ((p2 & 1u) ? f.x : 0.f) - ((m2 & 1u) ? f.x : 0.f)
             + ((p2 & 2u) ? f.y : 0.f) - ((m2 & 2u) ? f.y : 0.f);
      }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[n] = acc * scale;
  }
}

// Ternary, unpack-once-to-registers then HFMA2. The unpack is done with 16-bit
// integer selects (no float ops), and the multiply-add runs on the FP16 pipe.
__global__ void k_gemv_t3_unpack(const uint32_t* __restrict__ P,
                                 const uint32_t* __restrict__ M,
                                 const __half*   __restrict__ x,
                                 float*          __restrict__ y,
                                 int N, int K, float scale) {
  extern __shared__ __half sx[];
  stage_x(x, sx, K);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int NW = blockDim.x >> 5, WPR = K / 32;
  for (int n = blockIdx.x * NW + warp; n < N; n += gridDim.x * NW) {
    const uint32_t* pr = P + (size_t)n * WPR;
    const uint32_t* mr = M + (size_t)n * WPR;
    float acc = 0.f;
    for (int g = lane; g < WPR; g += 32) {
      const uint32_t pw = pr[g], mw = mr[g];
      const __half2* xg2 = reinterpret_cast<const __half2*>(sx + g * 32);
      #pragma unroll
      for (int j = 0; j < 16; ++j) {
        // build two half2 masks in the integer domain, then multiply
        uint32_t p2 = (pw >> (2 * j)) & 0x3u;
        uint32_t m2 = (mw >> (2 * j)) & 0x3u;
        // 0x3C00 == 1.0h ; combine two lanes into one packed half2 constant
        uint32_t pmask = 0, mmask = 0;
        if (p2 & 1u) pmask |= 0x00003C00u;
        if (p2 & 2u) pmask |= 0x3C000000u;
        if (m2 & 1u) mmask |= 0x00003C00u;
        if (m2 & 2u) mmask |= 0x3C000000u;
        __half2 xv = xg2[j];
        __half2 pv, mv;
        memcpy(&pv, &pmask, 4);
        memcpy(&mv, &mmask, 4);
        __half2 t = __hsub2(__hmul2(xv, pv), __hmul2(xv, mv));
        float2 f = __half22float2(t);
        acc += f.x + f.y;
      }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[n] = acc * scale;
  }
}

// ---- ternary GEMV producing fp16 output, used inside the forward pass ----
// Same maths as k_gemv_t3_warp but writes __half so the next layer can consume
// it without an fp32 round trip.
__global__ void k_gemv_t3_h(const uint32_t* __restrict__ P,
                            const uint32_t* __restrict__ M,
                            const __half*   __restrict__ x,
                            const __half*   __restrict__ scale,
                            __half*         __restrict__ y,
                            int N, int K) {
  extern __shared__ __half sx[];
  stage_x(x, sx, K);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int NW = blockDim.x >> 5, WPR = K / 32;
  for (int n = blockIdx.x * NW + warp; n < N; n += gridDim.x * NW) {
    const uint32_t* pr = P + (size_t)n * WPR;
    const uint32_t* mr = M + (size_t)n * WPR;
    float acc = 0.f;
    for (int g = lane; g < WPR; g += 32) {
      const uint32_t pw = pr[g], mw = mr[g];
      const __half2* xg2 = reinterpret_cast<const __half2*>(sx + g * 32);
      #pragma unroll
      for (int j = 0; j < 16; ++j) {
        uint32_t p2 = (pw >> (2 * j)) & 0x3u;
        uint32_t m2 = (mw >> (2 * j)) & 0x3u;
        float2 f = __half22float2(xg2[j]);
        acc += ((p2 & 1u) ? f.x : 0.f) - ((m2 & 1u) ? f.x : 0.f)
             + ((p2 & 2u) ? f.y : 0.f) - ((m2 & 2u) ? f.y : 0.f);
      }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[n] = __float2half(acc * __half2float(scale[n]));
  }
}

void launch_gemv_t3_h(const uint32_t* dP, const uint32_t* dM, const __half* dx,
                      const __half* dscale, __half* dy, int N, int K, cudaStream_t s) {
  int th = 256, nw = th / 32, bl = (N + nw - 1) / nw, sm = K * 2;
  set_smem(k_gemv_t3_h, sm);
  k_gemv_t3_h<<<bl, th, sm, s>>>(dP, dM, dx, dscale, dy, N, K);
}

// ================================================================== launchers
void launch_gemv_int4_naive(const uint32_t* dW, const __half* dx, float* dy,
                            int N, int K, float scale, cudaStream_t s) {
  int th = 256; k_gemv_int4_naive<<<(N + th - 1) / th, th, 0, s>>>(dW, dx, dy, N, K, scale);
}
void launch_gemv_int4_warp(const uint32_t* dW, const __half* dx, float* dy,
                           int N, int K, float scale, cudaStream_t s) {
  int th = 256, nw = th / 32, bl = (N + nw - 1) / nw, sm = K * 2;
  set_smem(k_gemv_int4_warp, sm);
  k_gemv_int4_warp<<<bl, th, sm, s>>>(dW, dx, dy, N, K, scale);
}
void launch_gemv_f16(const __half* dW, const __half* dx, float* dy,
                     int N, int K, cudaStream_t s) {
  int th = 256, nw = th / 32, bl = (N + nw - 1) / nw, sm = K * 2;
  set_smem(k_gemv_f16, sm);
  k_gemv_f16<<<bl, th, sm, s>>>(dW, dx, dy, N, K);
}
void launch_gemv_t3_warp(const uint32_t* dP, const uint32_t* dM, const __half* dx,
                         float* dy, int N, int K, float scale, cudaStream_t s) {
  int th = 256, nw = th / 32, bl = (N + nw - 1) / nw, sm = K * 2;
  set_smem(k_gemv_t3_warp, sm);
  k_gemv_t3_warp<<<bl, th, sm, s>>>(dP, dM, dx, dy, N, K, scale);
}
void launch_gemv_t3_unpack(const uint32_t* dP, const uint32_t* dM, const __half* dx,
                           float* dy, int N, int K, float scale, cudaStream_t s) {
  int th = 256, nw = th / 32, bl = (N + nw - 1) / nw, sm = K * 2;
  set_smem(k_gemv_t3_unpack, sm);
  k_gemv_t3_unpack<<<bl, th, sm, s>>>(dP, dM, dx, dy, N, K, scale);
}

// ================================================================== reference
void gemv_int4_reference(const uint32_t* W, const __half* x, float* y, int N, int K, float scale) {
  for (int n = 0; n < N; ++n) {
    const uint32_t* row = W + (size_t)n * (K / 8);
    double acc = 0;
    for (int g = 0; g < K / 8; ++g) {
      uint32_t w = row[g];
      for (int j = 0; j < 8; ++j)
        acc += (double)((int)((w >> (4 * j)) & 0xF) - 8) * (double)__half2float(x[g * 8 + j]);
    }
    y[n] = (float)(acc * scale);
  }
}
void gemv_t3_reference(const uint32_t* P, const uint32_t* M, const __half* x, float* y, int N, int K, float scale) {
  for (int n = 0; n < N; ++n) {
    const uint32_t* pr = P + (size_t)n * (K / 32);
    const uint32_t* mr = M + (size_t)n * (K / 32);
    double acc = 0;
    for (int k = 0; k < K; ++k) {
      int w = (int)((pr[k / 32] >> (k & 31)) & 1u) - (int)((mr[k / 32] >> (k & 31)) & 1u);
      acc += (double)w * (double)__half2float(x[k]);
    }
    y[n] = (float)(acc * scale);
  }
}
void gemv_f16_reference(const __half* W, const __half* x, float* y, int N, int K) {
  for (int n = 0; n < N; ++n) {
    double acc = 0;
    for (int k = 0; k < K; ++k) acc += (double)__half2float(W[(size_t)n * K + k]) * (double)__half2float(x[k]);
    y[n] = (float)acc;
  }
}

// ================================================================= L2 window
size_t query_l2_persist_max() {
  int v = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxPersistingL2CacheSize, 0));
  return (size_t)v;
}

size_t L2Window::engage(void* p, size_t n, cudaStream_t s) {
  size_t cap = query_l2_persist_max();
  int    win = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&win, cudaDevAttrMaxAccessPolicyWindowSize, 0));
  size_t take = n; if (take > (size_t)win) take = (size_t)win;
  if (take > cap) take = cap;
  if (take == 0) return 0;

  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, take));

  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow.base_ptr  = p;
  attr.accessPolicyWindow.num_bytes = take;
  attr.accessPolicyWindow.hitRatio  = 1.0f;
  attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
  CUDA_CHECK(cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr));

  ptr = p; bytes = take; active = true;
  return take;
}

void L2Window::release(cudaStream_t s) {
  if (!active) return;
  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow.base_ptr  = nullptr;
  attr.accessPolicyWindow.num_bytes = 0;
  attr.accessPolicyWindow.hitRatio  = 0.f;
  attr.accessPolicyWindow.hitProp   = cudaAccessPropertyNormal;
  attr.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;
  cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
  cudaCtxResetPersistingL2Cache();
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
  active = false;
}

} // namespace canticle

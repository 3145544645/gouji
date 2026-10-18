
// ============================================================================
//  CANTICLE full-model training kernels (fp32 activations, ternary weights).
// ============================================================================
#include "canticle/ftrain.h"
#include "canticle/ops.h"
#include <cuda_fp16.h>
#include <cmath>

namespace canticle {

#define TPB 256
static inline int nblk(int n) { return (n + TPB - 1) / TPB; }

// ---------------------------------------------------------------- rmsnorm fwd
// one block per row; writes the inverse-rms factor for the backward pass
__global__ void k_rms_fwd(const float* __restrict__ x, const float* __restrict__ w,
                          float* __restrict__ y, float* __restrict__ rrms,
                          int L, int d, float eps) {
  const int row = blockIdx.x;
  if (row >= L) return;
  const float* xr = x + (size_t)row * d;
  float ss = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) ss += xr[i] * xr[i];
  #pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
  __shared__ float red[32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f; int nw = (blockDim.x + 31) >> 5;
    for (int i = 0; i < nw; ++i) t += red[i];
    red[0] = rsqrtf(t / (float)d + eps);
  }
  __syncthreads();
  const float r = red[0];
  if (threadIdx.x == 0) rrms[row] = r;
  for (int i = threadIdx.x; i < d; i += blockDim.x)
    y[(size_t)row * d + i] = xr[i] * r * w[i];
}

// ---------------------------------------------------------------- rmsnorm bwd
// dx[i] = r*(dy[i]*w[i] - xhat[i]*(1/d)*sum_j dy[j]*w[j]*xhat[j])
__global__ void k_rms_bwd(const float* __restrict__ x, const float* __restrict__ w,
                          const float* __restrict__ dy, const float* __restrict__ rrms,
                          float* __restrict__ dx, float* __restrict__ dw,
                          int L, int d) {
  const int row = blockIdx.x;
  if (row >= L) return;
  const float* xr = x + (size_t)row * d;
  const float* dyr = dy + (size_t)row * d;
  float* dxr = dx + (size_t)row * d;
  const float r = rrms[row];

  float dot = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float xh = xr[i] * r;
    dot += dyr[i] * w[i] * xh;
    atomicAdd(&dw[i], dyr[i] * xh);          // norm gain grad
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, o);
  __shared__ float red[32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = dot;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f; int nw = (blockDim.x + 31) >> 5;
    for (int i = 0; i < nw; ++i) t += red[i];
    red[0] = t / (float)d;
  }
  __syncthreads();
  const float inv = red[0];
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float xh = xr[i] * r;
    dxr[i] += r * (dyr[i] * w[i] - xh * inv);
  }
}

// ------------------------------------------------------------------ rope fwd
__global__ void k_rope_fwd(float* __restrict__ q, float* __restrict__ k,
                           int L, int H, int hd, float theta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int half = hd / 2;
  int total = L * H * half;
  if (idx >= total) return;
  int i = idx % half;
  int h = (idx / half) % H;
  int t = idx / (half * H);
  float freq = 1.0f / powf(theta, (2.0f * i) / (float)hd);
  float ang = t * freq, c = cosf(ang), s = sinf(ang);
  float* qh = q + ((size_t)t * H + h) * hd;
  float* kh = k + ((size_t)t * H + h) * hd;
  float q0 = qh[2*i], q1 = qh[2*i+1];
  qh[2*i]   = q0*c - q1*s;
  qh[2*i+1] = q0*s + q1*c;
  float k0 = kh[2*i], k1 = kh[2*i+1];
  kh[2*i]   = k0*c - k1*s;
  kh[2*i+1] = k0*s + k1*c;
}

__global__ void k_rope_bwd(float* __restrict__ dq, float* __restrict__ dk,
                           int L, int H, int hd, float theta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int half = hd / 2;
  int total = L * H * half;
  if (idx >= total) return;
  int i = idx % half;
  int h = (idx / half) % H;
  int t = idx / (half * H);
  float freq = 1.0f / powf(theta, (2.0f * i) / (float)hd);
  float ang = t * freq, c = cosf(ang), s = sinf(ang);
  float* qh = dq + ((size_t)t * H + h) * hd;
  float* kh = dk + ((size_t)t * H + h) * hd;
  float q0 = qh[2*i], q1 = qh[2*i+1];
  qh[2*i]   =  q0*c + q1*s;
  qh[2*i+1] = -q0*s + q1*c;
  float k0 = kh[2*i], k1 = kh[2*i+1];
  kh[2*i]   =  k0*c + k1*s;
  kh[2*i+1] = -k0*s + k1*c;
}

// --------------------------------------------------------------- causal attn
// One block (one warp) per (head, query position). L <= 256 assumed.
__global__ void k_attn_fwd(const float* __restrict__ q, const float* __restrict__ k,
                           const float* __restrict__ v, float* __restrict__ att,
                           float* __restrict__ probs,   // [H x L x L]
                           int L, int H, int hd, float scale) {
  const int h = blockIdx.x / L;
  const int t = blockIdx.x % L;
  if (h >= H) return;
  const int lane = threadIdx.x & 31;
  float* pr = probs + ((size_t)h * L + t) * L;

  float mx = -1e30f;
  for (int s = 0; s <= t; ++s) {
    const float* qh = q + ((size_t)t * H + h) * hd;
    const float* kh = k + ((size_t)s * H + h) * hd;
    float d = 0.f;
    for (int i = lane; i < hd; i += 32) d += qh[i] * kh[i];
    #pragma unroll
    for (int o = 16; o; o >>= 1) d += __shfl_down_sync(0xffffffffu, d, o);
    d = __shfl_sync(0xffffffffu, d, 0) * scale;
    if (lane == 0) pr[s] = d;
    mx = fmaxf(mx, d);
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  mx = __shfl_sync(0xffffffffu, mx, 0);

  // Only lane 0 owns pr[]. Every other lane must not read pr[] until lane 0
  // has finished writing it -- __shfl_sync synchronises the *value transfer*
  // but not the *memory* lane 0 just wrote. Without these barriers lane 1 can
  // read a stale raw score (tens of units large) instead of a probability,
  // which overflows the accumulated attention output to inf.
  float sum = 0.f;
  for (int s = 0; s <= t; ++s) { if (lane == 0) { float e = __expf(pr[s] - mx); pr[s] = e; sum += e; } }
  sum = __shfl_sync(0xffffffffu, sum, 0);
  if (lane == 0) for (int s = 0; s <= t; ++s) pr[s] /= sum;
  __syncwarp();

  float* ah = att + ((size_t)t * H + h) * hd;
  for (int i = lane; i < hd; i += 32) {
    float acc = 0.f;
    for (int s = 0; s <= t; ++s) acc += pr[s] * v[((size_t)s * H + h) * hd + i];
    ah[i] = acc;
  }
}

// attention backward: given datt, produce dq, dk, dv and the dprobs slab
__global__ void k_attn_bwd(const float* __restrict__ q, const float* __restrict__ k,
                           const float* __restrict__ v, const float* __restrict__ att,
                           const float* __restrict__ probs,
                           const float* __restrict__ datt,
                           float* __restrict__ dprobs,   // [H x L x L] scratch
                           float* __restrict__ dq, float* __restrict__ dk,
                           float* __restrict__ dv,
                           int L, int H, int hd, float scale) {
  const int h = blockIdx.x / L;
  const int t = blockIdx.x % L;
  if (h >= H) return;
  const int lane = threadIdx.x & 31;
  const float* pr = probs + ((size_t)h * L + t) * L;
  float* dpr = dprobs + ((size_t)h * L + t) * L;
  const float* dat = datt + ((size_t)t * H + h) * hd;

  // dv[s] += p[s] * datt
  for (int s = 0; s <= t; ++s) {
    for (int i = lane; i < hd; i += 32)
      atomicAdd(&dv[((size_t)s * H + h) * hd + i], pr[s] * dat[i]);
  }
  // dp[s] = datt . v[s]
  for (int s = 0; s <= t; ++s) {
    const float* vh = v + ((size_t)s * H + h) * hd;
    float d = 0.f;
    for (int i = lane; i < hd; i += 32) d += dat[i] * vh[i];
    #pragma unroll
    for (int o = 16; o; o >>= 1) d += __shfl_down_sync(0xffffffffu, d, o);
    if (lane == 0) dpr[s] = d;
  }
  // softmax backward (lane 0 owns dpr; barrier before any other lane reads it)
  float dotp = 0.f;
  for (int s = 0; s <= t; ++s) { if (lane == 0) dotp += dpr[s] * pr[s]; }
  for (int s = 0; s <= t; ++s) {
    if (lane == 0) dpr[s] = pr[s] * (dpr[s] - dotp);
  }
  __syncwarp();
  // dq, dk
  for (int s = 0; s <= t; ++s) {
    const float* qh = q + ((size_t)t * H + h) * hd;
    const float* kh = k + ((size_t)s * H + h) * hd;
    float gs = (lane == 0) ? dpr[s] * scale : 0.f;
    gs = __shfl_sync(0xffffffffu, gs, 0);
    for (int i = lane; i < hd; i += 32) {
      atomicAdd(&dq[((size_t)t * H + h) * hd + i], gs * kh[i]);
      atomicAdd(&dk[((size_t)s * H + h) * hd + i], gs * qh[i]);
    }
  }
}

// --------------------------------------------------------------- silu / mul
__global__ void k_silu_fwd(const float* __restrict__ g, const float* __restrict__ u,
                           float* __restrict__ h, float* __restrict__ sig,
                           int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = g[i];
  float s = 1.0f / (1.0f + __expf(-x));
  sig[i] = s;
  h[i] = x * s * u[i];
}
__global__ void k_silu_bwd(const float* __restrict__ g, const float* __restrict__ u,
                           const float* __restrict__ sig, const float* __restrict__ dh,
                           float* __restrict__ dg, float* __restrict__ du, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = g[i], s = sig[i];
  // d(x*silu(x))/dx = s + x*s*(1-s)
  float ds = s + x * s * (1.0f - s);
  dg[i] += dh[i] * ds * u[i];
  du[i] += dh[i] * x * s;
}

// ------------------------------------------------------------------ add
__global__ void k_add_fwd(float* __restrict__ dst, const float* __restrict__ src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

// ------------------------------------------------------------------ cast
__global__ void k_cast_f16(const float* __restrict__ s, __half* __restrict__ d, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = __float2half(s[i]);
}

// ---------------------------------------------------- batched launchers
void fw_rmsnorm(const float* x, const float* w, float* y, float* rrms,
                int L, int d, float eps, cudaStream_t s) {
  k_rms_fwd<<<L, TPB, 0, s>>>(x, w, y, rrms, L, d, eps);
}
void bw_rmsnorm(const float* x, const float* w, const float* dy, const float* rrms,
                float* dx, float* dw, int L, int d, cudaStream_t s) {
  k_rms_bwd<<<L, TPB, 0, s>>>(x, w, dy, rrms, dx, dw, L, d);
}
void fw_rope(float* q, float* k, int L, int H, int hd, float theta, cudaStream_t s) {
  int total = L * H * (hd / 2);
  k_rope_fwd<<<nblk(total), TPB, 0, s>>>(q, k, L, H, hd, theta);
}
void bw_rope(float* dq, float* dk, int L, int H, int hd, float theta, cudaStream_t s) {
  int total = L * H * (hd / 2);
  k_rope_bwd<<<nblk(total), TPB, 0, s>>>(dq, dk, L, H, hd, theta);
}
void fw_attn(const float* q, const float* k, const float* v, float* att, float* probs,
             int L, int H, int hd, float scale, cudaStream_t s) {
  k_attn_fwd<<<L * H, 32, 0, s>>>(q, k, v, att, probs, L, H, hd, scale);
}
void bw_attn(const float* q, const float* k, const float* v, const float* att,
             const float* probs, const float* datt, float* dprobs,
             float* dq, float* dk, float* dv, int L, int H, int hd, float scale,
             cudaStream_t s) {
  k_attn_bwd<<<L * H, 32, 0, s>>>(q, k, v, att, probs, datt, dprobs, dq, dk, dv, L, H, hd, scale);
}
void fw_silu(const float* g, const float* u, float* h, float* sig, int n, cudaStream_t s) {
  k_silu_fwd<<<nblk(n), TPB, 0, s>>>(g, u, h, sig, n);
}
void bw_silu(const float* g, const float* u, const float* sig, const float* dh,
             float* dg, float* du, int n, cudaStream_t s) {
  k_silu_bwd<<<nblk(n), TPB, 0, s>>>(g, u, sig, dh, dg, du, n);
}
void add_inplace32(float* dst, const float* src, int n, cudaStream_t s) {
  k_add_fwd<<<nblk(n), TPB, 0, s>>>(dst, src, n);
}
void cast_f32_to_f16(const float* src, __half* dst, long n, cudaStream_t s) {
  k_cast_f16<<<nblk((int)n), TPB, 0, s>>>(src, dst, (int)n);
}

// ------------------------------------------------------- embedding (fp32)
__global__ void k_embed_fwd32(const float* __restrict__ emb, const int* __restrict__ ids,
                              float* __restrict__ x, int L, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L * d) return;
  int t = i / d, j = i % d;
  x[i] = emb[(size_t)ids[t] * d + j];
}

__global__ void k_copy_f32(float* __restrict__ dst, const float* __restrict__ src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

void embed_fwd32(const float* emb, const int* ids, float* x, int L, int d, cudaStream_t s) {
  k_embed_fwd32<<<nblk(L * d), TPB, 0, s>>>(emb, ids, x, L, d);
}
void copy_f32(float* dst, const float* src, int n, cudaStream_t s) {
  k_copy_f32<<<nblk(n), TPB, 0, s>>>(dst, src, n);
}

} // namespace canticle

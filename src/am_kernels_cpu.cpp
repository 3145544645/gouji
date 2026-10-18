// ===========================================================================
//  am_kernels_cpu.cpp -- the CPU half of the retrieval primitives.
//
//  WHY THIS FILE EXISTS
//  -------------------
//  The project's stated goal is a model that runs on ANY low-performance device.
//  Until this file existed, the entire thing required CUDA, which means the goal
//  was contradicted by the build itself. A 8 GB laptop GPU is not 'any device'.
//
//  WHAT IT IS, AND IS NOT
//  ---------------------
//  It is NOT a reimplementation of the model. am.cpp, chat.cpp and every am-* test
//  suite contain zero CUDA references; only the nine primitives below touch the
//  device. So the CPU port is nine functions, not a rewrite.
//
//  It is NOT fast. The CUDA path scans 13500 x 2048 ternary keys in 0.22 ms. This
//  one will be roughly an order of magnitude slower and that is fine: correctness
//  first, and the speed is measured rather than assumed (am-bench reports it).
//
//  THE ONE INTERESTING PART
//  ----------------------
//  Ternary scoring is popcount over 32 dimensions at a time. On a GPU that is one
//  instruction per warp lane; on a CPU it is __builtin_popcountll, also one
//  instruction. The ALGORITHM survives the port unchanged -- only the parallelism
//  across N is lost. That is why this port is tractable at all.
// ===========================================================================

#include "canticle/am.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace canticle {

namespace {

// The CPU path has no stream. The parameter is kept so the signatures match.
inline void noop_stream() {}

float half_to_float(uint16_t h) {
  // IEEE 754 binary16 -> binary32. Written out rather than using a hardware
  // intrinsic because the point of this file is portability.
  const uint32_t sign = (uint32_t)(h >> 15) & 0x1u;
  const uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
  const uint32_t mant = (uint32_t)h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign << 31;
    } else {
      // subnormal: renormalise
      uint32_t e = 0, m = mant;
      while ((m & 0x400u) == 0) { m <<= 1; ++e; }
      m &= 0x3FFu;
      bits = (sign << 31) | ((127 - 15 - e + 1) << 23) | (m << 13);
    }
  } else if (exp == 0x1F) {
    bits = (sign << 31) | 0x7F800000u | (mant << 13);
  } else {
    bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

inline uint16_t float_to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 31) & 0x1u;
  const int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0) return (uint16_t)(sign << 15);          // underflow to zero
  if (exp >= 0x1F) return (uint16_t)((sign << 15) | 0x7C00u);  // saturate
  return (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | (mant >> 13));
}

inline int popcount64(uint64_t v) {
#if defined(_MSC_VER)
  return (int)__popcnt64(v);
#else
  return __builtin_popcountll(v);
#endif
}

} // namespace

// ---- fp16 dot product ----------------------------------------------------
void am_dot(const uint16_t* K, const float* q, float* sim, int N, int D,
            void* /*stream*/) {
  for (int i = 0; i < N; ++i) {
    const uint16_t* row = K + (size_t)i * D;
    float acc = 0.f;
    for (int d = 0; d < D; ++d) acc += half_to_float(row[d]) * q[d];
    sim[i] = acc;
  }
}

void am_to_f16(const float* src, uint16_t* dst, long n, void* /*stream*/) {
  for (long i = 0; i < n; ++i) dst[i] = float_to_half(src[i]);
}

} // namespace canticle

// ---- f16 -> f32 widening -------------------------------------------------
//
//  model.cpp declares this extern and calls it to widen embedding tables. The
//  definition lives in kernels.cu, which the CPU build does not compile, so it is
//  provided here instead. Declared at global scope, not in namespace canticle,
//  because that is how the CUDA definition and every call site spell it.
void launch_f16_to_f32(const uint16_t* src, float* dst, int n, void*) {
  // Spelled out here rather than calling the anonymous-namespace helper above:
  // this function is at GLOBAL scope in the CUDA build, so a helper inside
  // namespace canticle would not be visible. The arithmetic is identical to
  // canticle::f16_to_f32 in common.h, which the tests exercise.
  for (int i = 0; i < n; ++i) {
    const uint16_t h = src[i];
    const uint32_t sign = (uint32_t)(h >> 15) & 0x1u;
    const uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
      bits = (sign << 31);
    } else if (exp == 0x1F) {
      bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
      bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &bits, 4);
    dst[i] = f;
  }
}

// A canticle:: face for the same function, so am_main.cpp can call it by the
// qualified name it uses in the CUDA build. One implementation, two names --
// cheaper than teaching every call site about the difference.
namespace canticle {
void launch_f16_to_f32(const uint16_t* src, float* dst, int n, void* s) {
  ::launch_f16_to_f32(src, dst, n, s);
}
} // namespace canticle

namespace canticle {

// ---- gathers -------------------------------------------------------------
void am_gather_f16(const uint16_t* src, const int* rows, uint16_t* dst, int m, int D,
                   void* /*stream*/) {
  for (int i = 0; i < m; ++i)
    std::memcpy(dst + (size_t)i * D, src + (size_t)rows[i] * D, (size_t)D * 2);
}

void am_gather_tern(const uint32_t* P, const uint32_t* M, const float* inv,
                    const int* rows, uint32_t* dP, uint32_t* dM, float* dInv,
                    int m, int WPR, void* /*stream*/) {
  for (int i = 0; i < m; ++i) {
    const int r = rows[i];
    std::memcpy(dP + (size_t)i * WPR, P + (size_t)r * WPR, (size_t)WPR * 4);
    std::memcpy(dM + (size_t)i * WPR, M + (size_t)r * WPR, (size_t)WPR * 4);
    dInv[i] = inv[r];
  }
}

// ---- ternary quantisation ------------------------------------------------
//
//  thr_frac is a FRACTION of the row's max |value|, not an absolute threshold.
//  That is what makes the encoding scale-free: the same sentence at two volumes
//  quantises identically. Getting this wrong (using an absolute threshold) is the
//  kind of defect that would pass on one corpus and fail on another.
void am_quantize_keys(const float* K, int N, int D, uint32_t* P, uint32_t* M,
                      float* invnorm, float thr_frac, void* /*stream*/) {
  const int WPR = (D + 31) / 32;   // words per row
  for (int i = 0; i < N; ++i) {
    const float* row = K + (size_t)i * D;
    float mx = 0.f;
    for (int d = 0; d < D; ++d) mx = std::max(mx, std::fabs(row[d]));
    const float thr = mx * thr_frac;
    float sq = 0.f;
    for (int w = 0; w < WPR; ++w) {
      uint32_t wp = 0, wm = 0;
      for (int b = 0; b < 32; ++b) {
        const int d = w * 32 + b;
        if (d >= D) break;
        const float v = row[d];
        if (v >  thr) { wp |= (1u << b); sq += 1.f; }
        else if (v < -thr) { wm |= (1u << b); sq += 1.f; }
      }
      P[(size_t)i * WPR + w] = wp;
      M[(size_t)i * WPR + w] = wm;
    }
    invnorm[i] = sq > 0.f ? 1.f / std::sqrt(sq) : 0.f;
  }
}

void am_quantize_query(const float* q, int D, uint32_t* P, uint32_t* M,
                       float thr_frac, float* nnz_out, void* /*stream*/) {
  const int WPR = (D + 31) / 32;
  float mx = 0.f;
  for (int d = 0; d < D; ++d) mx = std::max(mx, std::fabs(q[d]));
  const float thr = mx * thr_frac;
  float sq = 0.f;
  for (int w = 0; w < WPR; ++w) {
    uint32_t wp = 0, wm = 0;
    for (int b = 0; b < 32; ++b) {
      const int d = w * 32 + b;
      if (d >= D) break;
      const float v = q[d];
      if (v >  thr) { wp |= (1u << b); sq += 1.f; }
      else if (v < -thr) { wm |= (1u << b); sq += 1.f; }
    }
    P[w] = wp;
    M[w] = wm;
  }
  if (nnz_out) *nnz_out = sq;   // squared, matching the CUDA path's convention
}

// ---- ternary dot ---------------------------------------------------------
//
//  agree - disagree, scaled by sqrt(nnz_row * nnz_query). Exactly the GPU formula;
//  only the loop over N moves to the host.
void am_dot_ternary(const uint32_t* KP, const uint32_t* KM, const float* invnorm,
                    const uint32_t* QP, const uint32_t* QM,
                    float* sim, int N, int D, float qnnz, void* /*stream*/) {
  const int WPR = (D + 31) / 32;
  const float denom_q = std::sqrt(qnnz);
  for (int i = 0; i < N; ++i) {
    const uint32_t* rp = KP + (size_t)i * WPR;
    const uint32_t* rm = KM + (size_t)i * WPR;
    int agree = 0, disagree = 0;
    for (int w = 0; w < WPR; ++w) {
      const uint32_t ap = rp[w] & QP[w];
      const uint32_t am = rm[w] & QM[w];
      const uint32_t dp = rp[w] & QM[w];
      const uint32_t dm = rm[w] & QP[w];
      agree += popcount64((uint64_t)ap) + popcount64((uint64_t)am);
      disagree += popcount64((uint64_t)dp) + popcount64((uint64_t)dm);
    }
    const float denom = denom_q * (invnorm[i] > 0.f ? 1.f / invnorm[i] : 0.f);
    sim[i] = denom > 0.f ? (float)(agree - disagree) / denom : 0.f;
  }
}

// ---- softmax, backproject, l2 --------------------------------------------
void am_softmax(float* s, int N, float beta, void* /*stream*/) {
  if (N <= 0) return;
  float mx = s[0];
  for (int i = 1; i < N; ++i) mx = std::max(mx, s[i]);
  float sum = 0.f;
  for (int i = 0; i < N; ++i) { s[i] = std::exp((s[i] - mx) * beta); sum += s[i]; }
  if (sum > 0.f) for (int i = 0; i < N; ++i) s[i] /= sum;
}

void am_back(const uint16_t* K, const float* w, float* qout, int N, int D,
             void* /*stream*/) {
  for (int d = 0; d < D; ++d) qout[d] = 0.f;
  for (int i = 0; i < N; ++i) {
    const float wi = w[i];
    if (wi == 0.f) continue;
    const uint16_t* row = K + (size_t)i * D;
    for (int d = 0; d < D; ++d) qout[d] += wi * half_to_float(row[d]);
  }
}

void am_l2(float* v, int n, void* /*stream*/) {
  double sq = 0.0;
  for (int i = 0; i < n; ++i) sq += (double)v[i] * v[i];
  if (sq <= 0.0) return;
  const float inv = (float)(1.0 / std::sqrt(sq));
  for (int i = 0; i < n; ++i) v[i] *= inv;
}

} // namespace canticle

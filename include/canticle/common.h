#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

// CUDA headers are conditional. See the note in am.h: a CPU build must not
// require a toolkit, and these two includes were the only other place that did.
// CUDA_CHECK stays defined under CANTICLE_CPU so that call sites compile
// unchanged -- a portable build should not change the shape of the code it builds.
#ifdef CANTICLE_CPU
  // EVALUATE, do not sizeof. The first version used sizeof(x) to keep the macro
  // portable without running the expression -- which broke the moment a call site
  // passed a copy: sizeof(cudaMemcpyAsync(...)) is sizeof(void), and the compiler
  // reported "illegal sizeof operand" at eight lines that had nothing wrong with
  // them.
  //
  // On the host path the work is real (a memcpy, a stream no-op), so the right
  // thing is to run it and discard the result.
  #define CUDA_CHECK(x) do { (void)(x); } while(0)
  // The rest of the CUDA surface that call sites touch directly. On the host these
  // are plain memory operations: the device/host distinction has nothing to map to.
  #define cudaMemcpy(d, s, n, k)  (std::memcpy((d), (s), (n)), 0)
  #define cudaMemset(p, v, n)     (std::memset((p), (v), (n)), 0)
  #define cudaFree(p)             (std::free(p), 0)
#else
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
  #define CUDA_CHECK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
    std::fprintf(stderr,"[CUDA ERR] %s:%d  %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
    std::exit(1);} } while(0)
#endif


namespace canticle {

inline double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct DeviceInfo {
  char  name[256] = {0};
  int   sm_major = 0, sm_minor = 0;
  int   sm_count = 0;
  long long total_mem = 0;
  int   bus_width = 0;
  int   l2_bytes = 0;
  size_t l2_persist_max = 0;      // cudaDevAttrMaxPersistingL2CacheSize
  size_t access_policy_max = 0;   // cudaDevAttrMaxAccessPolicyWindowSize
  int   shmem_per_block = 0;
  int   regs_per_sm = 0;
  int   max_threads_per_sm = 0;
};

DeviceInfo query_device(int dev);

// fp16 conversion. The CUDA build uses the hardware intrinsics; the CPU build
// uses the same bit-twiddling that am_kernels_cpu.cpp already needs, factored
// here so there is one implementation rather than two that can drift apart.
#ifdef CANTICLE_CPU
inline uint16_t f32_to_f16(float f) {
  uint32_t x; std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 31) & 0x1u;
  const int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
  const uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0)  return (uint16_t)(sign << 15);
  if (exp >= 0x1F) return (uint16_t)((sign << 15) | 0x7C00u);
  return (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | (mant >> 13));
}
inline float f16_to_f32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h >> 15) & 0x1u;
  const uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
  const uint32_t mant = (uint32_t)h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) { bits = sign << 31; }
    else {
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
  float f; std::memcpy(&f, &bits, 4); return f;
}
#else
inline uint16_t f32_to_f16(float f) {
  __half h = __float2half(f);
  uint16_t u; std::memcpy(&u, &h, 2); return u;
}
inline float f16_to_f32(uint16_t u) {
  __half h; std::memcpy(&h, &u, 2); return __half2float(h);
}
// Overload guard: without this, __half silently converts to float and then
// truncates to uint16_t, producing garbage. Cost us one debugging round.
inline float f16_to_f32(__half h) { return __half2float(h); }
#endif

// Global aliases for the CUDA spellings, defined AFTER the conversion functions
// they wrap. Placing them earlier produced "not a member of canticle", because
// the wrappers referred to declarations that did not exist yet -- the same
// ordering mistake as an early return hiding later work, one scope smaller.
#ifdef CANTICLE_CPU
using cudaStream_t = void*;
using __half = uint16_t;
inline __half __float2half(float f) { return canticle::f32_to_f16(f); }
inline float  __half2float(__half h) { return canticle::f16_to_f32(h); }
#endif

} // namespace canticle

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
  std::fprintf(stderr,"[CUDA ERR] %s:%d  %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
  std::exit(1);} } while(0)

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

} // namespace canticle

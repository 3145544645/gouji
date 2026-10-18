
#include "canticle/common.h"
namespace canticle {
DeviceInfo query_device(int dev) {
  DeviceInfo d;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  std::snprintf(d.name, sizeof(d.name), "%s", p.name);
  d.sm_major = p.major; d.sm_minor = p.minor;
  d.sm_count = p.multiProcessorCount;
  d.total_mem = (long long)p.totalGlobalMem;
  d.bus_width = p.memoryBusWidth;
  d.l2_bytes = p.l2CacheSize;
  d.shmem_per_block = (int)p.sharedMemPerBlock;
  d.regs_per_sm = p.regsPerMultiprocessor;
  d.max_threads_per_sm = p.maxThreadsPerMultiProcessor;
  int v = 0;
  if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxPersistingL2CacheSize, dev) == cudaSuccess) d.l2_persist_max = (size_t)v;
  if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxAccessPolicyWindowSize, dev) == cudaSuccess) d.access_policy_max = (size_t)v;
  return d;
}
}

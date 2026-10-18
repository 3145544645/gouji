#pragma once
#include "canticle/common.h"

// ===========================================================================
//  HOST/D EVICE BUFFER, AND THE MEMCPY ENUMS.
//
//  GpuBuf is the ONE abstraction am_main.cpp uses to hold the key matrix. Its
//  name is now slightly wrong: under CANTICLE_CPU it is host memory. It keeps the
//  name so that the ~55 call sites in am_main.cpp do not change -- a portable
//  build should not reshape the code it builds.
//
//  The cudaMemcpyKind enumerators are likewise aliased to plain ints, because
//  am_main.cpp passes them straight through. Under the CPU build nothing reads
//  the direction: a host-to-host copy is a memcpy either way.
// ===========================================================================
#ifdef CANTICLE_CPU
  #include <cstdlib>
  #include <cstring>
  namespace canticle {
    using memcpy_kind_t = int;
    enum { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2,
           cudaMemcpyDeviceToDevice = 3 };
    // Host stand-ins for the async copies am_main.cpp issues.
    inline void can_memcpy_async(void* dst, const void* src, size_t n, int, void*) {
      std::memcpy(dst, src, n);
    }
  }
  #define cudaMemcpyAsync can_memcpy_async
  #define cudaStreamCreate(p) (*(p) = nullptr, 0)
  #define cudaStreamSynchronize(s) ((void)(s), 0)
#else
  namespace canticle { using memcpy_kind_t = cudaMemcpyKind; }
#endif

namespace canticle {

struct GpuBuf {
  void*  ptr = nullptr;
  size_t bytes = 0;

  void alloc(size_t b) {
    if (bytes >= b) return;                       // already big enough: keep it
    void* np = nullptr;
#ifdef CANTICLE_CPU
    np = std::malloc(b);
    if (!np) { std::fprintf(stderr, "[OOM] %zu bytes\n", b); std::exit(1); }
    if (ptr && bytes) std::memcpy(np, ptr, bytes);
    if (ptr) std::free(ptr);
#else
    CUDA_CHECK(cudaMalloc(&np, b));
    if (ptr && bytes) CUDA_CHECK(cudaMemcpy(np, ptr, bytes, cudaMemcpyDeviceToDevice));
    if (ptr) cudaFree(ptr);
#endif
    ptr = np; bytes = b;
  }

  // Ensure at least b bytes of capacity, preserving whatever is already there.
  void grow(size_t b) { alloc(b); }

  // Ensure room for n elements of elem bytes at a given current count.
  void ensure(size_t n, size_t elem) { alloc(n * elem); }

  void free_() {
#ifdef CANTICLE_CPU
    if (ptr) { std::free(ptr); ptr = nullptr; bytes = 0; }
#else
    if (ptr) { cudaFree(ptr); ptr = nullptr; bytes = 0; }
#endif
  }
  ~GpuBuf() { free_(); }
};

// ================= GEMV variants (route A, GPU only) ======================
#ifndef CANTICLE_CPU
void launch_gemv_int4_naive(const uint32_t* dW, const __half* dx, float* dy,
                            int N, int K, float scale, cudaStream_t s);
void launch_gemv_int4_warp(const uint32_t* dW, const __half* dx, float* dy,
                           int N, int K, float scale, cudaStream_t s);
void launch_gemv_f16(const __half* dW, const __half* dx, float* dy,
                     int N, int K, cudaStream_t s);
void launch_gemv_t3_h(const uint32_t* dP, const uint32_t* dM, const __half* dx,
                      const __half* dscale, __half* dy, int N, int K, cudaStream_t s);
void launch_gemv_t3_warp(const uint32_t* dP, const uint32_t* dM, const __half* dx,
                         float* dy, int N, int K, float scale, cudaStream_t s);
void launch_gemv_t3_unpack(const uint32_t* dP, const uint32_t* dM, const __half* dx,
                           float* dy, int N, int K, float scale, cudaStream_t s);

// ---- CPU references ----
void gemv_int4_reference(const uint32_t* W, const __half* x, float* y, int N, int K, float scale);
void gemv_t3_reference(const uint32_t* P, const uint32_t* M, const __half* x, float* y, int N, int K, float scale);
void gemv_f16_reference(const __half* W, const __half* x, float* y, int N, int K);

// ---- L2 persistence (Innovation #2) ----
// Pin [ptr, ptr+bytes) into the 32 MiB L2 so decode never touches DRAM again.
struct L2Window {
  void*  ptr = nullptr;
  size_t bytes = 0;
  size_t prev_limit = 0;
  bool   active = false;
  // returns the number of bytes actually reserved for persisting
  size_t engage(void* p, size_t n, cudaStream_t s);
  void   release(cudaStream_t s);
};
size_t query_l2_persist_max();

void copy_bandwidth_probe(float* out_gbps, size_t bytes, int iters, cudaStream_t s);
void fill_random(uint32_t* d, size_t words, uint32_t seed, cudaStream_t s);
void flush_l2(const void* d, size_t bytes, cudaStream_t s);


#endif // !CANTICLE_CPU

} // namespace canticle

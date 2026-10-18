#pragma once
#include "canticle/common.h"

namespace canticle {

// ---------------------------------------------------------------------------
//  A device buffer that can GROW WITHOUT LOSING ITS CONTENTS.
//
//  WHY THIS MATTERS FOR ONLINE LEARNING
//  ------------------------------------
//  The old alloc() freed the block before allocating a bigger one, which is fine
//  for a scratch buffer and fatal for a memory matrix: appending pattern N+1
//  would wipe patterns 1..N. That is precisely why this system could only learn
//  by full rebuild -- am-build regenerated the entire memory from scratch every
//  time, so nothing could be added at run time and nothing could be corrected.
//
//  grow() allocates the new block, copies the old contents, and only then frees
//  the old one. Capacity is monotonically increasing and never shrinks, so an
//  append never invalidates a pointer held elsewhere for longer than one call.
//
//  alloc() keeps its old semantics for genuine scratch buffers, but it no longer
//  silently discards data when the request fits inside the existing capacity.
// ---------------------------------------------------------------------------
struct GpuBuf {
  void*  ptr = nullptr;
  size_t bytes = 0;

  void alloc(size_t b) {
    if (bytes >= b) return;                       // already big enough: keep it
    void* np = nullptr;
    CUDA_CHECK(cudaMalloc(&np, b));
    if (ptr && bytes) CUDA_CHECK(cudaMemcpy(np, ptr, bytes, cudaMemcpyDeviceToDevice));
    if (ptr) cudaFree(ptr);
    ptr = np; bytes = b;
  }

  // Ensure at least b bytes of capacity, preserving whatever is already there.
  void grow(size_t b) { alloc(b); }

  // Ensure room for n elements of elem bytes at a given current count.
  void ensure(size_t n, size_t elem) { alloc(n * elem); }

  void free_() { if (ptr) { cudaFree(ptr); ptr = nullptr; bytes = 0; } }
  ~GpuBuf() { free_(); }
};

// ================= GEMV variants (N x K matrix . K-vector -> N vector) ======
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

} // namespace canticle

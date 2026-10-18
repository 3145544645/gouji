
// ============================================================================
//  CANTICLE - sweep harness
//
//  Question this answers: at what weight footprint does each format win?
//  On a 32 MiB L2 part, a small model is *instruction* bound while a large one
//  is *bandwidth* bound. The optimum format therefore depends on model size.
//  This sweep locates that crossover with real measurements, per DRAM read.
// ============================================================================
#include "canticle/common.h"
#include "canticle/kernels.h"
#include "canticle/quant.h"
#include <vector>
#include <algorithm>

using namespace canticle;

template <typename F>
static double timeit_f(F&& f, int warm, int iters, cudaStream_t s) {
  for (int i = 0; i < warm; ++i) f();
  CUDA_CHECK(cudaStreamSynchronize(s));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, s);
  for (int i = 0; i < iters; ++i) f();
  cudaEventRecord(e1, s);
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms / iters;
}

int sweep_run() {
  const int K = 4096;
  const int iters = 200;
  DeviceInfo d = query_device(0);

  cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));

  // A 256 MiB scratch arena, walked to evict everything useful from L2.
  void* flushbuf = nullptr;
  CUDA_CHECK(cudaMalloc(&flushbuf, 512ull * 1024 * 1024));
  CUDA_CHECK(cudaMemset(flushbuf, 3, 512ull * 1024 * 1024));

  GpuBuf dx, dy;
  dx.alloc(K * 2); dy.alloc(65536 * 4);
  CUDA_CHECK(cudaMemsetAsync(dx.ptr, 0x11, K * 2, st));

  std::printf("=== CANTICLE sweep: N x %d, L2=%.1f MiB, persist_max=%.1f MiB ===\n",
              K, d.l2_bytes / 1048576.0, d.l2_persist_max / 1048576.0);
  std::printf("%8s | %8s %8s %8s | %8s %8s %8s | %8s %8s %8s | %s\n",
              "N", "t3 MB", "i4 MB", "f16 MB",
              "t3 ms", "i4 ms", "f16 ms",
              "t3 GB/s", "i4 GB/s", "f16 GB/s", "winner");

  std::vector<int> Ns = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

  // worst-case GPU time per format at each size, so we size the arena once
  size_t maxwords_t3 = (size_t)65536 * (K / 32);
  GpuBuf dP, dM, dI4, dF16;
  dP.alloc(maxwords_t3 * 4); dM.alloc(maxwords_t3 * 4);
  dI4.alloc((size_t)65536 * (K / 8) * 4);
  dF16.alloc((size_t)65536 * K * 2);

  for (int N : Ns) {
    size_t t3b = (size_t)N * (K / 32) * 4 * 2;
    size_t i4b = (size_t)N * (K / 8) * 4;
    size_t f16b = (size_t)N * K * 2;

    // Fill weights with pseudo random data. Content does not matter for
    // speed, only for realism of the decode.
    fill_random((uint32_t*)dP.ptr,  (size_t)N * (K / 32), 0x1234u, st);
    fill_random((uint32_t*)dM.ptr,  (size_t)N * (K / 32), 0x5678u, st);
    fill_random((uint32_t*)dI4.ptr, (size_t)N * (K / 8),  0x9abcu, st);
    fill_random((uint32_t*)dF16.ptr, (size_t)N * K / 2,   0xdef0u, st);
    CUDA_CHECK(cudaStreamSynchronize(st));

    auto* Pp = (const uint32_t*)dP.ptr;
    auto* Mp = (const uint32_t*)dM.ptr;
    auto* Ip = (const uint32_t*)dI4.ptr;
    auto* Fp = (const __half*)dF16.ptr;
    auto* xp = (const __half*)dx.ptr;
    auto* yp = (float*)dy.ptr;

    // Cold measurement: evict L2 before every timed iteration, then time ONE
    // launch. This is the honest model of a decode step for weights that do
    // not fit in cache. (Warm/cached numbers are reported separately.)
    auto cold = [&](auto&& f) {
      for (int i = 0; i < 3; ++i) f();
      CUDA_CHECK(cudaStreamSynchronize(st));
      const int C = 100;
      double acc = 0;
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
      for (int i = 0; i < C; ++i) {
        flush_l2(flushbuf, 512ull * 1024 * 1024, st);
        CUDA_CHECK(cudaStreamSynchronize(st));
        cudaEventRecord(e0, st);
        f();
        cudaEventRecord(e1, st);
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        acc += ms;
      }
      cudaEventDestroy(e0); cudaEventDestroy(e1);
      return acc / C;
    };

    double t3 = cold([&]{ launch_gemv_t3_warp(Pp, Mp, xp, yp, N, K, 0.01f, st); });
    double i4 = cold([&]{ launch_gemv_int4_warp(Ip, xp, yp, N, K, 0.01f, st); });
    double f16 = cold([&]{ launch_gemv_f16(Fp, xp, yp, N, K, st); });

    double t3g = t3b * 1e-9 / (t3 * 1e-3);
    double i4g = i4b * 1e-9 / (i4 * 1e-3);
    double f16g = f16b * 1e-9 / (f16 * 1e-3);

    const char* win = "t3"; double best = t3;
    if (i4 < best) { best = i4; win = "int4"; }
    if (f16 < best) { best = f16; win = "f16"; }

    std::printf("%8d | %8.2f %8.2f %8.2f | %8.4f %8.4f %8.4f | %8.1f %8.1f %8.1f | %s\n",
                N, t3b / 1048576.0, i4b / 1048576.0, f16b / 1048576.0,
                t3, i4, f16, t3g, i4g, f16g, win);
    std::fflush(stdout);

    std::printf("{\"sweep\":{\"N\":%d,\"K\":%d,\"t3_ms\":%.5f,\"i4_ms\":%.5f,\"f16_ms\":%.5f,"
                "\"t3_mb\":%.3f,\"i4_mb\":%.3f,\"f16_mb\":%.3f,\"winner\":\"%s\"}}\n",
                N, K, t3, i4, f16, t3b / 1048576.0, i4b / 1048576.0, f16b / 1048576.0, win);
  }

  cudaFree(flushbuf);
  CUDA_CHECK(cudaStreamDestroy(st));
  return 0;
}

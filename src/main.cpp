
// ============================================================================
//  CANTICLE - command line entry point
//  Phase 0/1: benchmark harness producing machine readable numbers.
// ============================================================================
#include "canticle/common.h"
#include "canticle/kernels.h"
#include "canticle/quant.h"
#include <random>
#include <algorithm>
#include <functional>
#include <fstream>
#include <sstream>

using namespace canticle;

static void print_dev(const DeviceInfo& d) {
  std::printf("device        : %s\n", d.name);
  std::printf("compute       : sm_%d%d  SMs=%d\n", d.sm_major, d.sm_minor, d.sm_count);
  std::printf("memory        : %.2f GiB  bus=%d-bit\n", d.total_mem / 1073741824.0, d.bus_width);
  std::printf("l2 cache      : %.2f MiB\n", d.l2_bytes / 1048576.0);
  std::printf("l2 persist max: %.2f MiB   access window max: %.2f MiB\n",
              d.l2_persist_max / 1048576.0, d.access_policy_max / 1048576.0);
}

template <typename F>
static double timeit(F&& f, int iters, cudaStream_t s) {
  for (int i = 0; i < 3; ++i) f();
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

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
  return m;
}

#ifndef CANTICLE_CPU
static int cmd_bench(std::vector<std::string> args) {
  int N = 4096, K = 4096, iters = 300;
  size_t l2_target = 0;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--N"     && i + 1 < args.size()) N = std::atoi(args[i + 1].c_str());
    if (args[i] == "--K"     && i + 1 < args.size()) K = std::atoi(args[i + 1].c_str());
    if (args[i] == "--iters" && i + 1 < args.size()) iters = std::atoi(args[i + 1].c_str());
    if (args[i] == "--l2mb"  && i + 1 < args.size()) l2_target = (size_t)std::atoi(args[i + 1].c_str()) * 1048576;
  }
  K = (K + 31) / 32 * 32;

  DeviceInfo d = query_device(0);
  std::printf("=== CANTICLE bench : GEMV %dx%d  iters=%d ===\n", N, K, iters);
  print_dev(d);

  cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
  float copy_gbps = 0;
  copy_bandwidth_probe(&copy_gbps, 256ull * 1024 * 1024, 30, st);
  std::printf("\nmeasured copy BW : %.1f GB/s\n", copy_gbps);

  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> tern(-1, 1), i4(-8, 7);
  std::uniform_real_distribution<float> xd(-1.f, 1.f);

  Int4Matrix W4; W4.alloc(N, K); W4.scale = 0.0137f;
  TernaryMatrix Wt; Wt.alloc(N, K); Wt.scale = 0.0137f;
  std::vector<uint16_t> hf16((size_t)N * K);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      int q = i4(rng); W4.set(n, k, q);
      int t = tern(rng); Wt.set(n, k, t);
      hf16[(size_t)n * K + k] = f32_to_f16((float)q * W4.scale);
    }
  std::vector<uint16_t> hx(K);
  for (int k = 0; k < K; ++k) hx[k] = f32_to_f16(xd(rng));

  GpuBuf dW4, dP, dM, dF16, dx, dy;
  dW4.alloc(W4.W.size() * 4);
  dP.alloc(Wt.P.size() * 4);
  dM.alloc(Wt.M.size() * 4);
  dF16.alloc(hf16.size() * 2);
  dx.alloc(K * 2); dy.alloc(N * 4);
  CUDA_CHECK(cudaMemcpyAsync(dW4.ptr, W4.W.data(), W4.W.size() * 4, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(dP.ptr,  Wt.P.data(), Wt.P.size() * 4, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(dM.ptr,  Wt.M.data(), Wt.M.size() * 4, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(dF16.ptr, hf16.data(), hf16.size() * 2, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(dx.ptr, hx.data(), K * 2, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaStreamSynchronize(st));

  std::vector<float> ref4(N), refT(N), got(N);
  gemv_int4_reference(W4.W.data(), (const __half*)hx.data(), ref4.data(), N, K, W4.scale);
  gemv_t3_reference(Wt.P.data(), Wt.M.data(), (const __half*)hx.data(), refT.data(), N, K, Wt.scale);

  auto* W4p = (const uint32_t*)dW4.ptr;
  auto* Pp  = (const uint32_t*)dP.ptr;
  auto* Mp  = (const uint32_t*)dM.ptr;
  auto* Fp  = (const __half*)dF16.ptr;
  auto* xp  = (const __half*)dx.ptr;
  auto* yp  = (float*)dy.ptr;

  std::printf("\n--- correctness vs CPU reference ---\n");
  auto verify = [&](const std::vector<float>& ref, const char* tag) {
    CUDA_CHECK(cudaMemcpy(got.data(), dy.ptr, N * 4, cudaMemcpyDeviceToHost));
    double e = max_abs_diff(ref, got);
    double peak = 1e-9;
    for (float v : ref) peak = std::max(peak, (double)std::fabs(v));
    std::printf("  %-12s max_abs_err=%.3e  rel=%.3e  %s\n", tag, e, e / peak,
                e / peak < 2e-3 ? "PASS" : "FAIL");
    return e;
  };
  struct C { const char* tag; double err; };
  std::vector<C> errs;

  launch_gemv_int4_naive(W4p, xp, yp, N, K, W4.scale, st); CUDA_CHECK(cudaStreamSynchronize(st));
  errs.push_back({"int4-naive", verify(ref4, "int4-naive")});
  launch_gemv_int4_warp(W4p, xp, yp, N, K, W4.scale, st); CUDA_CHECK(cudaStreamSynchronize(st));
  errs.push_back({"int4-warp", verify(ref4, "int4-warp")});
  launch_gemv_f16(Fp, xp, yp, N, K, st); CUDA_CHECK(cudaStreamSynchronize(st));
  errs.push_back({"f16-hfma2", verify(ref4, "f16-hfma2")});
  launch_gemv_t3_warp(Pp, Mp, xp, yp, N, K, Wt.scale, st); CUDA_CHECK(cudaStreamSynchronize(st));
  errs.push_back({"t3-warp", verify(refT, "t3-warp")});
  launch_gemv_t3_unpack(Pp, Mp, xp, yp, N, K, Wt.scale, st); CUDA_CHECK(cudaStreamSynchronize(st));
  errs.push_back({"t3-unpack", verify(refT, "t3-unpack")});

  auto err_of = [&](const char* t) { for (auto& c : errs) if (!std::strcmp(c.tag, t)) return c.err; return 0.0; };

  // ---------------- throughput ----------------
  std::printf("\n--- throughput ---\n");
  std::printf("%-12s %10s %11s %10s %10s %8s\n", "kernel", "ms/iter", "GB/s eff", "% copy", "GFLOP/s", "bits/W");
  struct Row { std::string name; double ms, gbps, gf; size_t wb; double err; };
  std::vector<Row> rows;
  auto run = [&](const char* name, auto&& fn, size_t wb, double err) {
    double ms = timeit(fn, iters, st);
    double gbps = (double)wb * 1e-9 / (ms * 1e-3);
    double gf = 2.0 * N * K * 1e-9 / (ms * 1e-3);
    rows.push_back({name, ms, gbps, gf, wb, err});
    std::printf("%-12s %10.4f %11.1f %9.1f%% %10.1f %8.2f\n",
                name, ms, gbps, 100.0 * gbps / copy_gbps, gf, 8.0 * wb / ((double)N * K));
  };

  run("int4-naive", [&]{ launch_gemv_int4_naive(W4p, xp, yp, N, K, W4.scale, st); }, W4.weight_bytes(), err_of("int4-naive"));
  run("int4-warp",  [&]{ launch_gemv_int4_warp (W4p, xp, yp, N, K, W4.scale, st); }, W4.weight_bytes(), err_of("int4-warp"));
  run("f16-hfma2",  [&]{ launch_gemv_f16(Fp, xp, yp, N, K, st); },                    (size_t)N * K * 2, err_of("f16-hfma2"));
  run("t3-warp",    [&]{ launch_gemv_t3_warp  (Pp, Mp, xp, yp, N, K, Wt.scale, st); }, Wt.weight_bytes(), err_of("t3-warp"));
  run("t3-unpack",  [&]{ launch_gemv_t3_unpack(Pp, Mp, xp, yp, N, K, Wt.scale, st); }, Wt.weight_bytes(), err_of("t3-unpack"));

  // ---------------- Innovation #2: L2 residency A/B ----------------
  size_t persist = query_l2_persist_max();
  std::printf("\n--- innovation #2: L2 persistence A/B (dev max %.1f MiB) ---\n", persist / 1048576.0);
  size_t take = l2_target ? l2_target : persist;
  if (take > 0 && take <= persist) {
    // cold: no persistence window
    double ms_cold = timeit([&]{ launch_gemv_t3_warp(Pp, Mp, xp, yp, N, K, Wt.scale, st); }, iters, st);
    L2Window win;
    size_t got_bytes = win.engage(dP.ptr, take, st);
    for (int i = 0; i < 20; ++i) launch_gemv_t3_warp(Pp, Mp, xp, yp, N, K, Wt.scale, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    double ms_hot = timeit([&]{ launch_gemv_t3_warp(Pp, Mp, xp, yp, N, K, Wt.scale, st); }, iters, st);
    double bpw = 8.0 * Wt.weight_bytes() / ((double)N * K);
    double gbps_cold = (double)Wt.weight_bytes() * 1e-9 / (ms_cold * 1e-3);
    double gbps_hot  = (double)Wt.weight_bytes() * 1e-9 / (ms_hot * 1e-3);
    std::printf("  pinned %.2f MiB into L2\n", got_bytes / 1048576.0);
    std::printf("  t3-warp cold : %.4f ms  (%.1f GB/s eff)\n", ms_cold, gbps_cold);
    std::printf("  t3-warp hot  : %.4f ms  (%.1f GB/s eff)  speedup %.3fx\n",
                ms_hot, gbps_hot, ms_cold / ms_hot);
    win.release(st);
  } else {
    std::printf("  skipped (no persisting capacity)\n");
  }

  // ---------------- JSON ----------------
  std::printf("\n--- json ---\n{");
  std::printf("\"phase\":\"p1\",\"N\":%d,\"K\":%d,\"copy_gbps\":%.1f,", N, K, copy_gbps);
  std::printf("\"kernels\":[");
  for (size_t i = 0; i < rows.size(); ++i)
    std::printf("%s{\"name\":\"%s\",\"ms\":%.4f,\"gbps\":%.1f,\"pct_copy\":%.1f,\"gflops\":%.1f,\"weight_bytes\":%zu,\"max_err\":%.3e}",
                i ? "," : "", rows[i].name.c_str(), rows[i].ms, rows[i].gbps,
                100.0 * rows[i].gbps / copy_gbps, rows[i].gf, rows[i].wb, rows[i].err);
  std::printf("]}\n");

  CUDA_CHECK(cudaStreamDestroy(st));
  return 0;
}

#else  // CANTICLE_CPU

// Route A (GEMV benchmarks) is not built on the host path: it exists to measure
// bandwidth-bound kernels, and a CPU implementation would measure the host memory
// bus instead, which is a different question. The command stays REGISTERED so that
// "canticle bench" gives an answer rather than "unknown command" -- an error that
// says what is missing is worth more than one that pretends the command never
// existed.
static int cmd_bench(std::vector<std::string>) {
  std::printf("bench is not available in this build.\n\n");
  std::printf("  This binary was compiled WITHOUT CUDA (CANTICLE_CPU).\n");
  std::printf("  Route A benchmarks measure GPU memory bandwidth; there is nothing\n");
  std::printf("  equivalent to measure on the host.\n\n");
  std::printf("  What DOES work here: route B -- am-test, am-eval, am-bench,\n");
  std::printf("  am-realcclinc, am-chat and the rest of the dialogue suite.\n\n");
  std::printf("  To build the GPU path: cmake -DCANTICLE_FORCE_CPU=OFF\n");
  return 2;
}

#endif  // CANTICLE_CPU

#ifdef CANTICLE_CPU
// One explanation for every route-A command, instead of six near-identical ones.
// Called from the dispatch below; the message names what IS available, because
// "not built" without a next step just moves the confusion.
static int not_built(const char* what) {
  std::printf("%s is not available in this build.\n\n", what);
  std::printf("  This binary was compiled WITHOUT CUDA (CANTICLE_CPU), so it carries\n");
  std::printf("  route B only: the associative-memory dialogue and tool-routing line.\n\n");
  std::printf("  Works here: am-test am-source am-eval am-bench am-chat am-para\n");
  std::printf("              am-realclinc am-mutate fingerprint criteria\n\n");
  std::printf("  For the GPU path: cmake -DCANTICLE_FORCE_CPU=OFF\n");
  return 2;
}
#endif

int main(int argc, char** argv) {
  // Unbuffered stdout: a crash mid-run must not swallow the progress log.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::vector<std::string> a(argv + 1, argv + argc);
  if (a.empty() || a[0] == "help" || a[0] == "--help" || a[0] == "-h") {
    // The help text listed only two of the seventeen commands, so the whole
    // associative-memory path was undiscoverable from the binary itself.
    std::printf(
      "canticle - pure C++/CUDA ternary mini LLM\n"
      "usage: canticle <command> [options]\n\n"
      "  -- associative memory (the working dialogue + tool path) --\n"
      "  am-build  [--data N] [--am FILE]        one-shot Hebbian write of the memory\n"
      "  am-eval   [--data N] [--diag]           held-out generalisation score\n"
      "  am-bench                                retrieval latency and bandwidth\n"
      "  am-ask    --prompt \"TEXT\"               retrieve the best-matching episode\n"
      "  am-chat   [--prompt \"TEXT\"] [--debug]    full loop: retrieve, then EXECUTE the tool\n"
      "  am-test                                 unit tests for encoder + payload extraction\n\n"
      "  -- ternary kernels and training --\n"
      "  sweep                                   format crossover scan, cold L2 (0.5MB..268MB)\n"
      "  bench     [--N n --K k --iters i]       GEMV A/B: ternary vs int4 vs fp16\n"
      "  ste       [--steps n --rq n]            straight-through ternary learning gate\n"
      "  train     [--steps n --seq n --data n]  full-model ternary training\n"
      "  smoke                                   model forward + decode throughput\n\n"
      "  -- diagnostics --\n"
      "  dev                                     GPU attributes and L2 budget\n"
      "  align                                   train-vs-inference forward agreement\n"
      "  tokdiag                                 tokenizer save/load round trip\n"
      "  dump                                    print packed training sequences\n"
      "  chat      [--prompt \"TEXT\"] <ckpt>      transformer chat (known-limited; see STATUS.md)\n\n"
      "See STATUS.md for measured numbers and known limits.\n");
    return 1;
  }
  if (a[0] == "dev") { print_dev(query_device(0)); return 0; }
  if (a[0] == "bench") return cmd_bench(std::vector<std::string>(a.begin() + 1, a.end()));
#ifdef CANTICLE_CPU
  // Route A commands (dev diagnostics, kernels, training) are not linked on the
  // host path. Rather than wrap each dispatch in its own #ifdef -- six copies of
  // the same explanation -- one predicate names the set and one message says why.
  {
    // Only REAL command names. An invented one ("am-build-model") was in this list
    // for one build: am_source scans main.cpp for "am-" strings to check that every
    // suite has a mutation, found the invented name, and failed the gate. The
    // scanner cannot tell a command from a string that looks like one, so the list
    // must not contain anything that is not a command.
    static const char* kNotBuilt[] = {"diag","sweep","diag2","dump","tokdiag",
                                     "align","train","ftrain","smoke","ste",
                                     nullptr};
    for (int i = 0; kNotBuilt[i]; ++i) {
      if (a[0] == kNotBuilt[i]) {
        std::printf("%s is not available in this build.\n\n", a[0].c_str());
        std::printf("  This binary was compiled WITHOUT CUDA (CANTICLE_CPU).\n");
        std::printf("  It carries route B only: the associative-memory dialogue and\n");
        std::printf("  tool-routing line.\n\n");
        std::printf("  Available: am-test am-source am-eval am-bench am-chat\n");
        std::printf("             am-realclinc am-mutate fingerprint criteria\n\n");
        std::printf("  For the GPU path: cmake -DCANTICLE_FORCE_CPU=OFF\n");
        return 2;
      }
    }
  }
#else
  if (a[0] == "diag")  { extern int diag_run(); return diag_run(); }
  if (a[0] == "sweep") { extern int sweep_run(); return sweep_run(); }
  if (a[0] == "diag2") { extern int diag2_run(); return diag2_run(); }
  if (a[0] == "dump")  { extern int dump_run(); return dump_run(); }
  if (a[0] == "tokdiag") { extern int tokdiag_run(); return tokdiag_run(); }
  if (a[0] == "align") { extern int align_run(); return align_run(); }
#endif
  if (a[0] == "am-test") { extern int amtest_run(); return amtest_run(); }
  if (a[0] == "am-learntest") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_learn_run(int, char**);
    return am_learn_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-realclinc") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_realclinc_run(int, char**);
    return am_realclinc_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-entries") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_entries_run(int, char**);
    return am_entries_run((int)rest.size(), rest.data());
  }
  if (a[0] == "fingerprint") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_fingerprint_run(int, char**);
    return am_fingerprint_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-mutate") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_mutate_run(int, char**);
    return am_mutate_run((int)rest.size(), rest.data());
  }
  if (a[0] == "criteria") {
    // Print the distilled rules. They are the thing to read before writing code,
    // and am-source fails if the file is missing or truncated -- so this command
    // and that check point at the same artefact instead of two copies of it.
    std::ifstream f("CRITERIA.md", std::ios::binary);
    if (!f) {
      std::printf("CRITERIA.md not found. Run from the project root.\n");
      return 1;
    }
    std::stringstream ss; ss << f.rdbuf();
    std::printf("%s", ss.str().c_str());
    return 0;
  }
  if (a[0] == "am-source") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_source_run(int, char**);
    return am_source_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-turns") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_turns_run(int, char**);
    return am_turns_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-inject") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_inject_run(int, char**);
    return am_inject_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-hunt") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_hunt_run(int, char**);
    return am_hunt_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-atomic") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_atomic_run(int, char**);
    return am_atomic_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-generalize") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_generalize_run(int, char**);
    return am_generalize_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-precedence") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_precedence_run(int, char**);
    return am_precedence_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-unsupported") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_unsupported_run(int, char**);
    return am_unsupported_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-realcase") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_realcase_run(int, char**);
    return am_realcase_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-real") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_real_run(int, char**);
    return am_real_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-multi-act") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_multiact_run(int, char**);
    return am_multiact_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-collapse") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_collapse_run(int, char**);
    return am_collapse_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-induce") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_induce_run(int, char**);
    return am_induce_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-fixtest") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_fix_run(int, char**);
    return am_fix_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-harvest") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_run(int, char**);
    return am_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-learn") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_run(int, char**);
    return am_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-e2e") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_e2e_run(int, char**);
    return am_e2e_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-sim") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_sim_run(int, char**);
    return am_sim_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-sem") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_sem_run(int, char**);
    return am_sem_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-multi") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_multi_run(int, char**);
    return am_multi_run((int)rest.size(), rest.data());
  }
  if (a[0] == "am-para") {
    std::vector<char*> rest;
    for (size_t i = 1; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_para_run(int, char**);
    return am_para_run((int)rest.size(), rest.empty() ? nullptr : rest.data());
  }
  if (a[0] == "am-build" || a[0] == "am-eval" || a[0] == "am-ask" ||
      a[0] == "am-chat"  || a[0] == "am-bench") {
    std::vector<char*> rest;
    for (size_t i = 0; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
    extern int am_run(int, char**);
    return am_run((int)rest.size(), rest.data());
  }
  if (a[0] == "smoke") {
#ifndef CANTICLE_CPU
    extern int smoke_model_run(); return smoke_model_run();
#else
    return not_built("smoke");
#endif
  }
  if (a[0] == "train") {
    std::vector<char*> rest;
    for (size_t i = 1; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
#ifndef CANTICLE_CPU
    extern int full_train_run(int, char**);
    return full_train_run((int)rest.size(), rest.empty() ? nullptr : rest.data());
#else
    return not_built("train");
#endif
  }
  if (a[0] == "ste") {
    std::vector<char*> rest;
    for (size_t i = 1; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
#ifndef CANTICLE_CPU
    extern int train_run(int, char**);
    return train_run((int)rest.size(), rest.empty() ? nullptr : rest.data());
#else
    return not_built("ste");
#endif
  }
  if (a[0] == "chat") {
    std::vector<char*> rest;
    for (size_t i = 1; i < a.size(); ++i) rest.push_back(const_cast<char*>(a[i].c_str()));
#ifdef CANTICLE_CPU
    // "chat" drives the neural model (route A), which is GPU-only. The command
    // stays registered so the answer is an explanation rather than "unknown
    // command" -- a difference that matters when someone is deciding whether
    // their build is broken or merely incomplete.
    std::printf("chat is not available in this build.\n\n");
    std::printf("  This binary was compiled WITHOUT CUDA (CANTICLE_CPU).\n");
    std::printf("  \"chat\" runs the neural model, which needs a GPU.\n\n");
    std::printf("  The dialogue/tool-routing line does work here:\n");
    std::printf("    canticle am-chat --prompt \"read the file C:/tmp/a.txt\"\n");
    std::printf("    canticle am-test / am-eval / am-bench\n");
    return 2;
#else
    extern int chat_run(int, char**);
    return chat_run((int)rest.size(), rest.empty() ? nullptr : rest.data());
#endif
  }
  std::fprintf(stderr, "unknown command: %s\n", a[0].c_str());
  return 1;
}


// ============================================================================
//  CANTICLE-AM : associative memory core, GPU retrieval + host encoder.
//
//  FRAMEWORK (not a transformer, deliberately):
//
//    write : one pass, Hebbian, zero gradients, zero epochs
//    read  : nearest attractor, optionally refined by Hopfield cleanup
//
//  The GPU holds the memory matrix and does the retrieval; the encoder runs on
//  the host because it is a hash of the text and has nothing to learn. Keeping
//  the encoder off the GPU is not a compromise -- it is the point: there are no
//  parameters to train, so there is nothing to diverge or to evaluate
//  inconsistently with a training forward.
// ============================================================================
#include "canticle/am.h"
#include "canticle/lex.h"
#include "canticle/learn.h"
#include "canticle/induce.h"
#include "canticle/multiact.h"
// infer_intent is declared in am.h; the refusal rule below consults it to decide
// whether the structure layer already resolved this request.
#include <set>
#include <map>
#include "canticle/ckpt.h"
#include "canticle/data.h"
#include "canticle/kernels.h"
// ops.h (route A: embedding / attention / lm_head) is NOT included here: it was a
// dead include, and am_main.cpp calls none of its functions (verified by grepping
// every symbol ops.h declares). Removing it is what lets this file build without
// CUDA -- and am_main.cpp is the heart of route B, so the CPU port depends on it.
#ifndef CANTICLE_CPU
#include "canticle/ops.h"
#endif
#include "canticle/context.h"   // licences: path / expression / command
#ifndef CANTICLE_CPU
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#else
// On the host path there is no half type and no stream. Rather than rewrite the
// ~55 call sites below, the names are defined to mean what they mean on the GPU:
// a 16-bit key element, and a handle nothing waits on. char*, not void*, because
// some call sites take sizeof of an expression involving the stream type and
// sizeof(void) is ill-formed.
using __half = uint16_t;
using cudaStream_t = void*;
// QUALIFIED. f32_to_f16 / f16_to_f32 live in namespace canticle (common.h); these
// aliases are at global scope, so the unqualified names do not resolve. The error
// appears at the alias definition but reads like a missing include.
inline __half __float2half(float f) { return canticle::f32_to_f16(f); }
inline float __half2float(__half h) { return canticle::f16_to_f32(h); }
#endif
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <iostream>
#include <sstream>

namespace canticle {

#ifdef CANTICLE_CPU
// Provided by am_kernels_cpu.cpp. Declared inside the namespace because that is
// where the CUDA definition lives (train_ops.cu opens namespace canticle and does
// not close it before the function), so both builds see the same qualified name.
void launch_f16_to_f32(const half_t* src, float* dst, int n, stream_t s);
#endif

// 2048, NOT the hybrid 2208. A measured decision.
//
// Appending 160 lexical dimensions broke the ternary path even when the lexical
// weight was zero: the quantisation threshold is thr_frac * mean|x| over the
// WHOLE vector, so 160 appended zeros lowered the mean, lowered the threshold,
// and let more noise dimensions through. Held-out accuracy fell 100% -> 94.59%
// on the bucketed path and 100% -> 99.78% on the raw path.
//
// Combined with am-e2e showing the embedding changed no unseen-synonym
// behaviour, the honest conclusion is that this width should stay 2048.
// The hybrid code (hybrid_key, LexModel, HYB_W_LEX) is retained so the tradeoff
// can be re-measured rather than re-argued; see the note in hybrid.cpp.
static constexpr int   AM_D     = HYB_GRAM_DIM;
static float           AM_BETA  = 18.0f;
static int             AM_STEPS = 0;      // 0 = plain nearest neighbour

// ---------------------------------------------------------------- GPU kernels
// ---------------------------------------------------------------- memory
struct AmMem {
  int D = AM_D, N = 0;
  GpuBuf K;      // fp16 keys on the device -- half the bytes, same ranking
  // ---- ternary mirror of the key matrix (gap C) ----
  // Same rows, 2 bits per dimension instead of 16, scored by popcount. Kept
  // alongside the fp16 copy so the two can be compared on identical data
  // rather than on two separate builds.
  GpuBuf KP, KM, KINV;      // [N][D/32] planes + [N] reciprocal norms
  bool   ternary_ready = false;
  float  tern_thr = 0.25f;  // fraction of mean|x| used as the ternary threshold
  GpuBuf q, qb, sim, w, arg;

  void build_ternary() {
    if (N <= 0 || !K.ptr) return;
    const int WPR = D / 32;
    KP.alloc((size_t)N * WPR * 4);
    KM.alloc((size_t)N * WPR * 4);
    KINV.alloc((size_t)N * 4);
    // widen the fp16 keys once, quantise from the fp32 copy, then free it
    GpuBuf wide; wide.alloc((size_t)N * D * 4);
#ifdef CANTICLE_CPU
    // Provided by am_kernels_cpu.cpp under CANTICLE_CPU; by train_ops.cu otherwise.
    canticle::launch_f16_to_f32((const __half*)K.ptr, (float*)wide.ptr, N * D, st);
#else
    launch_f16_to_f32((const __half*)K.ptr, (float*)wide.ptr, N * D, st);
#endif
    am_quantize_keys((const float*)wide.ptr, N, D, (uint32_t*)KP.ptr,
                     (uint32_t*)KM.ptr, (float*)KINV.ptr, tern_thr, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    ternary_ready = true;
  }
  cudaStream_t st = nullptr;
  std::vector<std::string> prompts, replies;

  void ensure_stream() { if (!st) CUDA_CHECK(cudaStreamCreate(&st)); }

  // NOTE ON A REAL BUG THIS FIXES
  // GpuBuf::alloc frees the old block before allocating a bigger one. The first
  // version called reserve(idx+1) inside store(), so every growth step threw
  // away every key written so far and the memory silently kept only the last
  // few patterns -- which looked exactly like "the encoder has no
  // discrimination". reserve_all() is now called once for the full capacity.
  void reserve_all(int n) {
    ensure_stream();
    if (n <= 0) return;
    K.alloc((size_t)n * D * 2);      // fp16
    sim.alloc((size_t)n * 4);
    w.alloc((size_t)n * 4);
    q.alloc((size_t)D * 4);
    qb.alloc((size_t)D * 4);
    arg.alloc(4);
  }

  // Bucket index: for each tool family, the row numbers whose reply uses it.
  //
  // WHY: re-ranking the nearest K candidates failed even with perfect intent
  // inference, because "enumerate files under C:/tmp" has almost no trigram
  // overlap with a stored "list the directory ..." prompt -- no list_dir row
  // ever made it into the top-24. Restricting the SEARCH SPACE instead of
  // re-ranking its output is what actually fixes it: once we know the text is
  // about a directory, only directory rows are worth comparing against.
  // ---------------------------------------------------------------------------
  //  Bucket count and its invariant.
  //
  //  Buckets are indexed by Intent, but NOT every Intent gets a bucket. Only the
  //  five that correspond to a real tool can be searched, plus the catch-all:
  //
  //      ReadFile 0   ListDir 1   WriteFile 2   RunShell 3   Calc 4
  //      Unknown 5    (also used for Unsupported)
  //
  //  WHY THE COUNT IS THE HAZARD: adding Intent::Unsupported made the enum seven
  //  values wide while this array stayed at six. Every append with that intent then
  //  wrote one element past the end -- and because the rows are written to the GPU
  //  first, the symptom was not a crash at the write but "invalid argument" from a
  //  later kernel, three functions away from the cause.
  //
  //  Unsupported deliberately shares the Unknown bucket rather than getting its
  //  own: an unsupported operation is refused BEFORE retrieval, so it never needs
  //  to be searched. Giving it a bucket would only create another index to keep in
  //  step with the enum.
  //
  //  N_BUCKETS is used everywhere instead of a literal, so the next enum value
  //  cannot silently reintroduce this.
  // ---------------------------------------------------------------------------
  //  Bucket count, and the rule that keeps it honest.
  //
  //  The enum now reads: Unknown 0, ReadFile 1, ListDir 2, WriteFile 3, RunShell 4,
  //  Calc 5, Stat 6, Search 7, MkDir 8, Copy 9, Unsupported 10.
  //
  //  The count is now kIntentBucketCount in am.h, computed from the enum, so this
  //  comment is documentation rather than a second source of truth.
  //
  //  Every value EXCEPT Unsupported has a bucket, and the bucket index equals the
  //  enum value. Unsupported maps to Unknown because it is refused before any
  //  search runs -- it never needs to be findable.
  //
  //  Defect 49 was exactly this going wrong: the enum grew by one and the array did
  //  not, so every write went one element past the end and surfaced as an
  //  "invalid argument" from an unrelated kernel three calls later. bucket_of() and
  //  this constant exist so the next addition cannot repeat it.
  // ---------------------------------------------------------------------------
  // Derived, not declared. See kIntentBucketCount in am.h: a hand-written count
  // here is what produced defect 49, and the silent-fallback version that replaced
  // it merely moved the symptom.
  static constexpr int N_BUCKETS = kIntentBucketCount;
  static_assert(N_BUCKETS >= 10, "every tool intent must have a bucket index");
  static int bucket_of(Intent i) {
    if (i == Intent::Unsupported) return (int)Intent::Unknown;
    const int v = (int)i;
    return (v >= 0 && v < N_BUCKETS) ? v : (int)Intent::Unknown;
  }
  std::vector<int> bucket[N_BUCKETS];

  void rebuild_buckets() {
    for (int i = 0; i < N_BUCKETS; ++i) bucket[i].clear();
    // The name -> bucket mapping is READ FROM THE TOOL TABLE, not restated here.
    //
    // This was the third hand-maintained name->something chain in the codebase; the
    // first two were the dispatch chain and the whitelist (defect 73), and defect 70
    // was a fourth chain of the same shape that had silently omitted four tools.
    // The failure mode when this one falls behind is quieter than those: a tool
    // absent here is still stored, but under Unknown, so a structural query never
    // looks in its bucket and only retrieval by raw similarity can find it.
    for (int i = 0; i < (int)replies.size(); ++i) {
      const std::string t = tool_name_of(replies[i]);
      Intent in = Intent::Unknown;
      for (const ToolSpec& s : tool_specs())
        if (t == s.name) { in = (Intent)s.intent_id; break; }
      bucket[bucket_of(in)].push_back(i);
    }
  }
  static std::string tool_name_of(const std::string& s) {
    size_t a = s.find("<|tool|>");
    if (a == std::string::npos) return "";
    size_t b = s.find(' ', a);
    return s.substr(a + 8, (b == std::string::npos ? s.size() : b) - (a + 8));
  }

  // ---------------------------------------------------------------------------
  //  ONLINE APPEND.
  //
  //  This is the capability the architecture was built for and never exposed:
  //  Hebbian writing touches only the new row, so a pattern can be added in
  //  constant time with no rebuild and no gradient. Everything below was already
  //  true of the design; what was missing was that GpuBuf::alloc wiped the matrix
  //  on growth (see kernels.h) and that no command existed to append.
  //
  //  Cost per pattern: one encode on the host, one D-wide fp16 write, and one
  //  ternary quantisation of that single row. The existing N rows are untouched,
  //  so the cost does not grow with N.
  // ---------------------------------------------------------------------------
  bool append(const std::vector<float>& key, const std::string& prompt,
              const std::string& reply, std::string* err = nullptr) {
    ensure_stream();
    if ((int)key.size() != D) {
      if (err) *err = "key dimension mismatch";
      return false;
    }
    const int idx = (int)prompts.size();

    // -------------------------------------------------------------------
    //  EVERY buffer sized by N must be grown here, not just the key matrix.
    //
    //  The first version grew K, KP, KM and KINV but left sim[] and w[] at the
    //  capacity reserve_all() gave them. The append succeeded, the memory
    //  reported N+1 patterns, and the very next query died with
    //  "invalid argument" from a memcpy reading N*4 bytes out of a buffer sized
    //  for N-1. A growth path has to walk the whole set of N-dependent buffers;
    //  missing one turns a data bug into an opaque runtime error one call later.
    // -------------------------------------------------------------------
    const size_t needN = (size_t)(idx + 1) * 4;
    if (sim.bytes < needN) sim.alloc(needN);
    if (w.bytes   < needN) w.alloc(needN);

    // ---- grow the fp16 key matrix, preserving rows 0..idx-1 ----
    const size_t need = (size_t)(idx + 1) * D * 2;
    if (K.bytes < need) K.alloc(need);      // alloc now copies, never discards
    std::vector<__half> h(D);
    for (int i = 0; i < D; ++i) h[i] = __float2half(key[i]);
    CUDA_CHECK(cudaMemcpyAsync((uint8_t*)K.ptr + (size_t)idx * D * 2, h.data(),
                               (size_t)D * 2, cudaMemcpyHostToDevice, st));

    // ---- grow and write the ternary mirror for THIS ROW ONLY ----
    const int WPR = D / 32;
    if (ternary_ready) {
      const size_t tneed = (size_t)(idx + 1) * WPR * 4;
      if (KP.bytes < tneed) KP.alloc(tneed);
      if (KM.bytes < tneed) KM.alloc(tneed);
      if (KINV.bytes < (size_t)(idx + 1) * 4) KINV.alloc((size_t)(idx + 1) * 4);
      GpuBuf wide, wp, wm, wi;
      wide.alloc((size_t)D * 4);
      wp.alloc((size_t)WPR * 4);
      wm.alloc((size_t)WPR * 4);
      wi.alloc(4);
      CUDA_CHECK(cudaMemcpyAsync(wide.ptr, key.data(), (size_t)D * 4,
                                 cudaMemcpyHostToDevice, st));
      am_quantize_keys((const float*)wide.ptr, 1, D, (uint32_t*)wp.ptr,
                       (uint32_t*)wm.ptr, (float*)wi.ptr, tern_thr, st);
      CUDA_CHECK(cudaMemcpyAsync((uint8_t*)KP.ptr + (size_t)idx * WPR * 4, wp.ptr,
                                 (size_t)WPR * 4, cudaMemcpyDeviceToDevice, st));
      CUDA_CHECK(cudaMemcpyAsync((uint8_t*)KM.ptr + (size_t)idx * WPR * 4, wm.ptr,
                                 (size_t)WPR * 4, cudaMemcpyDeviceToDevice, st));
      CUDA_CHECK(cudaMemcpyAsync((uint8_t*)KINV.ptr + (size_t)idx * 4, wi.ptr,
                                 4, cudaMemcpyDeviceToDevice, st));
    }

    prompts.push_back(prompt);
    replies.push_back(reply);
    N = (int)prompts.size();

    // -----------------------------------------------------------------------
    //  BUCKET BY THE INTENT THE PROMPT WILL INFER, NOT BY THE REPLY'S TOOL.
    //
    //  This is the defect that made every learned correction invisible.
    //
    //  Retrieval narrows to bucket[infer_intent(query)] and searches only inside
    //  it. The old code filed an appended row under the bucket of the tool in its
    //  REPLY -- correct for the synthesised corpus, where prompt and reply always
    //  agree, and wrong for exactly the case learning exists to handle:
    //
    //      query "read the file C:/Windows"  -> infers ReadFile
    //      reply  list_dir {...}             -> filed under ListDir
    //
    //  The row was stored, persisted, and reported by every counter, then never
    //  looked at again, because the search went to the ReadFile bucket and the row
    //  sat in the ListDir one. A corrected answer whose inferred intent differs
    //  from its tool -- which is the DEFINITION of a correction -- could never be
    //  retrieved.
    //
    //  Filing by the prompt's inferred intent puts the row where the search will
    //  actually look. Rows from the base corpus are unaffected: there the two
    //  agree, so the bucket is the same either way. rebuild_buckets() still uses
    //  the reply's tool for the stored corpus, and that remains consistent because
    //  no stored prompt needs correcting.
    // -----------------------------------------------------------------------
    const Intent in = infer_intent(prompt);
    bucket[bucket_of(in)].push_back(idx);

    CUDA_CHECK(cudaStreamSynchronize(st));
    return true;
  }

  // ---------------------------------------------------------------------------
  //  Hybrid scoring.
  //
  //  The memory is keyed on [gram | lexical]. Retrieval first shortlists by the
  //  learned-lexical part only -- that is the part with measured semantic
  //  structure (margin 0.4890, 21/90 inversions against the gram encoder's
  //  84/90) -- and then re-scores that shortlist with the full hybrid key, so
  //  surface detail decides between candidates that mean the same thing.
  //
  //  Scoring in two stages rather than one dot product over the concatenation is
  //  a deliberate choice: a single dot would let a strong surface match beat a
  //  correct semantic match even when the two mean different things.
  // ---------------------------------------------------------------------------
  bool use_lex = false;

  void store_batch(const std::vector<std::vector<float>>& keys,
                   const std::vector<std::string>& ps,
                   const std::vector<std::string>& rs) {
    const int n = (int)keys.size();
    reserve_all(n);
    std::vector<float> flat((size_t)n * D);
    for (int i = 0; i < n; ++i)
      std::memcpy(flat.data() + (size_t)i * D, keys[i].data(), (size_t)D * 4);
    GpuBuf tmp; tmp.alloc(flat.size() * 4);
    CUDA_CHECK(cudaMemcpyAsync(tmp.ptr, flat.data(), flat.size() * 4,
                               cudaMemcpyHostToDevice, st));
    am_to_f16((const float*)tmp.ptr, (__half*)K.ptr, (long)n * D, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    prompts = ps; replies = rs;
    N = n;
    rebuild_buckets();
    build_ternary();
  }

  void store(const std::vector<float>& key, const std::string& p, const std::string& r) {
    const int idx = (int)prompts.size();
    if ((size_t)(idx + 1) * D * 2 > K.bytes) {
      // growth reallocates K; the existing rows were already written, so a
      // naive reserve here would drop them. store_batch() is the safe path.
      std::fprintf(stderr, "am: use store_batch for growing memories\n");
      return;
    }
    GpuBuf tmp; tmp.alloc((size_t)D * 4);
    CUDA_CHECK(cudaMemcpyAsync(tmp.ptr, key.data(), (size_t)D * 4, cudaMemcpyHostToDevice, st));
    am_to_f16((const float*)tmp.ptr, (__half*)K.ptr + (size_t)idx * D, D, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    prompts.push_back(p); replies.push_back(r);
    N = (int)prompts.size();
  }

  // Similarity against a SUBSET of rows (intent-restricted search).
  //
  // WHY: re-ranking the nearest K candidates failed even with 100% intent
  // accuracy, because "enumerate files under C:/tmp" shares almost no trigrams
  // with a stored "list the directory ..." prompt -- no list_dir row ever
  // reached the top-24. Restricting the SEARCH SPACE works where re-ranking its
  // output cannot.
  int query_subset(const std::vector<float>& key, const std::vector<int>& rows, int k,
                   std::vector<std::pair<float,int>>& top) {
    top.clear();
    if (rows.empty()) return -1;
    ensure_stream();
    if (!q.ptr) q.alloc((size_t)D * 4);
    CUDA_CHECK(cudaMemcpyAsync(q.ptr, key.data(), (size_t)D * 4, cudaMemcpyHostToDevice, st));

    const int m = (int)rows.size();
    const int WPR = D / 32;

    // Row list goes to the device once; every gather below is device-side.
    GpuBuf drows; drows.alloc((size_t)m * 4);
    CUDA_CHECK(cudaMemcpyAsync(drows.ptr, rows.data(), (size_t)m * 4,
                               cudaMemcpyHostToDevice, st));

    GpuBuf sim2; sim2.alloc((size_t)m * 4);
    if (ternary_ready) {
      GpuBuf sp, sm, si, qp, qm, qn;
      sp.alloc((size_t)m * WPR * 4);
      sm.alloc((size_t)m * WPR * 4);
      si.alloc((size_t)m * 4);
      qp.alloc((size_t)WPR * 4);
      qm.alloc((size_t)WPR * 4);
      qn.alloc(4);
      am_gather_tern((const uint32_t*)KP.ptr, (const uint32_t*)KM.ptr,
                     (const float*)KINV.ptr, (const int*)drows.ptr,
                     (uint32_t*)sp.ptr, (uint32_t*)sm.ptr, (float*)si.ptr,
                     m, WPR, st);
      am_quantize_query((const float*)q.ptr, D, (uint32_t*)qp.ptr, (uint32_t*)qm.ptr,
                        tern_thr, (float*)qn.ptr, st);
      float qnnz = 1.f;
      CUDA_CHECK(cudaMemcpyAsync(&qnnz, qn.ptr, 4, cudaMemcpyDeviceToHost, st));
      am_dot_ternary((const uint32_t*)sp.ptr, (const uint32_t*)sm.ptr,
                     (const float*)si.ptr, (const uint32_t*)qp.ptr,
                     (const uint32_t*)qm.ptr, (float*)sim2.ptr, m, D, qnnz, st);
    } else {
      GpuBuf sub; sub.alloc((size_t)m * D * 2);
      am_gather_f16((const __half*)K.ptr, (const int*)drows.ptr, (__half*)sub.ptr,
                    m, D, st);
      am_dot((const __half*)sub.ptr, (const float*)q.ptr, (float*)sim2.ptr, m, D, st);
    }
    std::vector<float> s(m);
    CUDA_CHECK(cudaMemcpyAsync(s.data(), sim2.ptr, (size_t)m * 4, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));

    std::vector<int> idx(m);
    for (int i = 0; i < m; ++i) idx[i] = i;
    const int kk = std::min(k, m);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](int a, int b) { return s[a] > s[b]; });
    for (int i = 0; i < kk; ++i) top.push_back({s[idx[i]], rows[idx[i]]});
    return top[0].second;
  }

  // ---- ternary path: 8x less data, integer comparison ----
  int query_ternary(const std::vector<float>& key, int k,
                    std::vector<std::pair<float,int>>& top) {
    top.clear();
    if (N == 0) return -1;
    if (!ternary_ready) return query(key, k, top);
    ensure_stream();
    if (!q.ptr) q.alloc((size_t)D * 4);
    CUDA_CHECK(cudaMemcpyAsync(q.ptr, key.data(), (size_t)D * 4, cudaMemcpyHostToDevice, st));

    const int WPR = D / 32;
    GpuBuf QP, QM, QN;
    QP.alloc((size_t)WPR * 4);
    QM.alloc((size_t)WPR * 4);
    QN.alloc(4);
    am_quantize_query((const float*)q.ptr, D, (uint32_t*)QP.ptr, (uint32_t*)QM.ptr,
                      tern_thr, (float*)QN.ptr, st);
    float qnnz = 1.f;
    CUDA_CHECK(cudaMemcpyAsync(&qnnz, QN.ptr, 4, cudaMemcpyDeviceToHost, st));

    GpuBuf sim2; sim2.alloc((size_t)N * 4);
    am_dot_ternary((const uint32_t*)KP.ptr, (const uint32_t*)KM.ptr,
                   (const float*)KINV.ptr, (const uint32_t*)QP.ptr,
                   (const uint32_t*)QM.ptr, (float*)sim2.ptr, N, D, qnnz, st);
    std::vector<float> s(N);
    CUDA_CHECK(cudaMemcpyAsync(s.data(), sim2.ptr, (size_t)N * 4, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));

    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    const int kk = std::min(k, N);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](int a, int b) { return s[a] > s[b]; });
    for (int i = 0; i < kk; ++i) top.push_back({s[idx[i]], idx[i]});
    return top[0].second;
  }

  // Two-stage hybrid retrieval. Falls back to a plain full-key search when the
  // lexical embedding is unavailable.
  int query_hybrid(const std::vector<float>& key,
                   const std::vector<float>& lexkey,
                   int stage1_k, int k,
                   std::vector<std::pair<float,int>>& top);

  // returns best index; fills top[] with the k best (index, similarity)
  int query(const std::vector<float>& key, int k, std::vector<std::pair<float,int>>& top) {
    if (N == 0) return -1;
    ensure_stream();
    if (!q.ptr) q.alloc((size_t)D * 4);
    if (!qb.ptr) qb.alloc((size_t)D * 4);
    if (!arg.ptr) arg.alloc(4);
    CUDA_CHECK(cudaMemcpyAsync(q.ptr, key.data(), (size_t)D * 4, cudaMemcpyHostToDevice, st));

    for (int it = 0; it < AM_STEPS; ++it) {
      am_dot((const __half*)K.ptr, (const float*)q.ptr, (float*)sim.ptr, N, D, st);
      am_softmax((float*)sim.ptr, N, AM_BETA, st);
      am_back((const __half*)K.ptr, (const float*)sim.ptr, (float*)qb.ptr, N, D, st);
      am_l2((float*)qb.ptr, D, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      CUDA_CHECK(cudaMemcpyAsync(q.ptr, qb.ptr, (size_t)D * 4, cudaMemcpyDeviceToDevice, st));
    }

    am_dot((const __half*)K.ptr, (const float*)q.ptr, (float*)sim.ptr, N, D, st);
    std::vector<float> s(N);
    CUDA_CHECK(cudaMemcpyAsync(s.data(), sim.ptr, (size_t)N * 4, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));

    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    const int kk = std::min(k, N);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](int a, int b) { return s[a] > s[b]; });
    top.clear();
    for (int i = 0; i < kk; ++i) top.push_back({s[idx[i]], idx[i]});
    return idx[0];
  }

  bool save(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write("CAM2", 4);
    int32_t hdr[2] = {D, N};
    f.write((const char*)hdr, 8);
    std::vector<__half> kh((size_t)N * D);
    if (N) CUDA_CHECK(cudaMemcpy(kh.data(), K.ptr, kh.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> k(kh.size());
    for (size_t i = 0; i < kh.size(); ++i) k[i] = __half2float(kh[i]);
    f.write((const char*)k.data(), (std::streamsize)k.size() * 4);
    for (int i = 0; i < N; ++i) {
      uint32_t a = (uint32_t)prompts[i].size(), b = (uint32_t)replies[i].size();
      f.write((const char*)&a, 4); f.write(prompts[i].data(), a);
      f.write((const char*)&b, 4); f.write(replies[i].data(), b);
    }
    return true;
  }
  // ---------------------------------------------------------------------------
  //  Append-only sidecar.
  //
  //  Rewriting the 52 MB matrix on every learned pattern would cost more than
  //  the learning itself. Instead each new pattern is appended to a small
  //  "<memory>.delta" file, which is replayed at load time. The base file stays
  //  immutable, so a mistaken correction can be discarded by deleting the delta
  //  rather than by rebuilding the whole memory.
  //
  //  Record layout:  int32 n_prompt | bytes | int32 n_reply | bytes   (little endian)
  // ---------------------------------------------------------------------------
  static std::string delta_path(const std::string& base) { return base + ".delta"; }

  bool append_delta(const std::string& base, const std::string& prompt,
                    const std::string& reply) {
    std::ofstream f(delta_path(base), std::ios::binary | std::ios::app);
    if (!f) return false;
    const int32_t a = (int32_t)prompt.size(), b = (int32_t)reply.size();
    f.write((const char*)&a, 4); if (a) f.write(prompt.data(), a);
    f.write((const char*)&b, 4); if (b) f.write(reply.data(), b);
    return (bool)f;
  }

  // Replay every delta record into this memory. Returns how many were applied.
  int replay_delta(const std::string& base, AmEncoder& enc, LexModel& lex) {
    std::ifstream f(delta_path(base), std::ios::binary);
    if (!f) return 0;
    int applied = 0;
    while (true) {
      int32_t a = 0;
      f.read((char*)&a, 4);
      if (!f) break;
      std::string p(a, '\0'); if (a) f.read(&p[0], a);
      int32_t b = 0;
      f.read((char*)&b, 4);
      if (!f) break;
      std::string r(b, '\0'); if (b) f.read(&r[0], b);
      std::vector<float> g, l, key;
      enc.encode(p, g);
      lex.encode(p, l);
      hybrid_key(g, l, key);
      if (append(key, p, r)) applied++;
    }
    return applied;
  }

  // Load the base memory AND replay any learned patterns.
  //
  // Every entry point must go through this, not through load(). Forgetting the
  // replay is a SILENT failure: the memory loads, retrieval runs, and the answer
  // is simply the old one. It cost one debugging round to find, because the
  // append and the delta write both reported success.
  bool load_with_delta(const std::string& path, AmEncoder& enc, LexModel& lex) {
    if (!load(path)) return false;
    const int applied = replay_delta(path, enc, lex);
    if (applied > 0 && getenv("AM_VERBOSE"))
      std::printf("[delta] replayed %d learned patterns (%d -> %d total)\n",
                  applied, N - applied, N);
    return true;
  }

  bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char mg[4]; f.read(mg, 4);
    if (std::memcmp(mg, "CAM2", 4)) return false;
    int32_t hdr[2]; f.read((char*)hdr, 8);
    D = hdr[0]; N = hdr[1];
    reserve_all(N);
    std::vector<float> k((size_t)N * D);
    f.read((char*)k.data(), (std::streamsize)k.size() * 4);
    if (N) {
      GpuBuf tmp; tmp.alloc(k.size() * 4);
      CUDA_CHECK(cudaMemcpy(tmp.ptr, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
      am_to_f16((const float*)tmp.ptr, (__half*)K.ptr, (long)k.size(), st);
      CUDA_CHECK(cudaStreamSynchronize(st));
    }
    prompts.resize(N); replies.resize(N);
    for (int i = 0; i < N; ++i) {
      uint32_t a = 0, b = 0;
      f.read((char*)&a, 4); prompts[i].resize(a); if (a) f.read(&prompts[i][0], a);
      f.read((char*)&b, 4); replies[i].resize(b); if (b) f.read(&replies[i][0], b);
    }
    rebuild_buckets();
    build_ternary();
    return true;
  }
};

// ---------------------------------------------------------------------------
//  Slot transfer.
//
//  The neural part picks WHICH episode; the exact payload is then copied from
//  the query rather than regenerated. A model re-emitting
//  "C:/work/gamma.txt" character by character will eventually drop one; a copy
//  is exact by construction.
// ---------------------------------------------------------------------------
// Copy the query's payload into the retrieved reply.
//
// The retrieved reply was written for a DIFFERENT payload ("53*456"), so we
// must locate that old payload inside it and substitute the new one. The match
// is done on a whitespace-stripped form, because the storage side spells it
// "53 * 456" in the prompt and "53*456" inside the JSON.
// ---------------------------------------------------------------------------
//  MULTI-ARGUMENT TOOL HOOK, at a fixed position.
//
//  Payload transfer has two kinds of case and they must run in a fixed order:
//
//      1. tools whose arguments ALL come from the query (copy: src + dst,
//         search: path + pattern partly). One substitution cannot serve these, so
//         they are rewritten here and returned.
//      2. every other tool: the stored reply's single payload is replaced.
//
//  Both attempts to add case 1 came AFTER an early return in case 2 and were
//  therefore dead code:
//      defect 51  search's pattern fix sat after "if (at != npos) return"
//      defect 57  copy's src/dst rewrite sat after "if (mpv == qp) return"
//  In both, the unit test for the extractor passed while the end-to-end answer was
//  wrong, because the extractor was correct and simply never called.
//
//  Rather than rely on remembering the rule, case 1 is now ONE function called
//  from ONE place, before any early return can exist. Adding the next
//  multi-argument tool means adding a branch to this function, where there is
//  nothing above it to shadow it.
//
//  Returns true when it handled the reply.
// ---------------------------------------------------------------------------
// Replace the value of a \"key\":\"...\" pair inside a JSON tool call.
//
// This three-step find (key, then closing quote, then replace) was written out
// four times: twice as a lambda for copy's src/dst, and twice inline for search's
// pattern and path. Four copies of a five-line invariant means four places to get
// the off-by-one wrong, and the failure mode is silent -- a wrong offset writes a
// truncated value that still parses as JSON.
//
// Stated once, here. Nothing about it is specific to any tool, which is precisely
// why it should not have been restated per tool.
static bool put_field(std::string* json, const char* key, const std::string& val) {
  const std::string kk = std::string("\"") + key + "\":\"";
  const size_t p = json->find(kk);
  if (p == std::string::npos) return false;
  const size_t vs = p + kk.size();
  const size_t ve = json->find('"', vs);
  if (ve == std::string::npos) return false;
  json->replace(vs, ve - vs, val);
  return true;
}

static bool rewrite_multi_arg(const std::string& q, const std::string& mr,
                              std::string* out) {
  // ---- copy: src and dst both come from the query ----
  //
  // Detected by the presence of BOTH field names rather than by the tool name. That
  // is deliberate: a field-name test cannot fall out of step with the tool table,
  // whereas a name test would need the same hand-synchronisation that produced
  // defects 70, 73 and 75.
  if (mr.find("\"src\"") != std::string::npos && mr.find("\"dst\"") != std::string::npos) {
    std::string s2, d2;
    *out = mr;
    if (!extract_copy_paths(q, &s2, &d2))
      return true;   // no destination: leave the skeleton so the tool reports it
    put_field(out, "src", s2);
    put_field(out, "dst", d2);
    return true;
  }
  // ---- search: BOTH path and pattern come from the query ----
  //
  //  The path was originally left to the generic substitution, on the reasoning
  //  that the stored template supplies it. That is wrong whenever the corpus and
  //  the query use different directories: measured 40 eval failures such as
  //      "search for config in C:/"  ->  {"path":"C:/tmp","pattern":"config"}
  //  -- the right needle searched in a directory the user never mentioned.
  //
  //  A two-argument tool has no argument the template can supply, so both are
  //  taken from the query, exactly as for copy.
  if (mr.find("\"pattern\"") != std::string::npos) {
    *out = mr;
    const std::string pat = extract_search_pattern(q);
    if (!pat.empty()) put_field(out, "pattern", pat);
    // The path: taken from the query too. extract_payload returns the path for a
    // search request, the needle having been consumed by extract_search_pattern.
    const std::string qpath = extract_payload(q);
    if (!qpath.empty()) put_field(out, "path", qpath);
    return true;
  }
  return false;
}

static std::string slot_transfer(const std::string& q,
                                 const std::string& mp, const std::string& mr) {
  // ---------------------------------------------------------------------------
  //  CASE 1 FIRST. Nothing may precede this.
  //
  //  See the note on rewrite_multi_arg: two separate defects came from putting
  //  multi-argument handling after an early return, so it now runs before any
  //  other logic in this function, including the guards below.
  // ---------------------------------------------------------------------------
  {
    std::string handled;
    if (rewrite_multi_arg(q, mr, &handled)) return handled;
  }

  // Canonicalise both sides first: the stored reply spells an expression
  // without spaces, so the substituted payload must match that convention.
  const std::string qp = normalise_payload(extract_payload(q));
  const std::string mpv = normalise_payload(extract_payload(mp));

  // -------------------------------------------------------------------------
  // ---- case 2: single-payload substitution ----
  //
  //  Everything below assumes the reply carries no more than one argument to
  //  replace. Multi-argument replies were already handled and returned above, so
  //  reaching this point means the generic path is the right one.
  if (mpv.empty() || qp.empty()) return mr;
  if (mpv == qp) return mr;

  // exact hit first
  size_t at = mr.find(mpv);
  if (at != std::string::npos) {
    std::string out = mr;
    out.replace(at, mpv.size(), qp);
    return out;
  }

  // whitespace-insensitive hit
  auto strip = [](const std::string& s) {
    std::string o; o.reserve(s.size());
    for (char c : s) if (c != ' ' && c != '\t') o.push_back(c);
    return o;
  };
  const std::string mpn = strip(mpv);
  std::string out = mr;
  const std::string outn = strip(out);
  size_t p = outn.find(mpn);
  if (p == std::string::npos) return out;

  // walk the unstripped string to the same logical position
  size_t real = 0, seen = 0;
  while (real < out.size() && seen < p) {
    if (out[real] != ' ' && out[real] != '\t') ++seen;
    ++real;
  }
  size_t rend = real, cnt = 0;
  while (rend < out.size() && cnt < mpn.size()) {
    if (out[rend] != ' ' && out[rend] != '\t') ++cnt;
    ++rend;
  }
  out.replace(real, rend - real, qp);
  return out;
}

// ---------------------------------------------------------------------------
//  Intent-aware retrieval.
//
//  Two signals, used for two different jobs:
//    * the n-gram vector says WHICH stored episode looks most like this text
//    * the structural intent says WHICH TOOL FAMILY the text belongs to
//
//  The n-gram vector alone put "what is in C:/tmp" on calc, because the words
//  "what is" are the strongest trigram overlap with any stored calc prompt.
//  Re-ranking the top-K candidates by intent affinity fixes that without
//  touching the encoder, and without weakening the n-gram signal inside a
//  family: among candidates that all route to list_dir, the most similar one
//  still wins, so the payload transfer stays accurate.
// ---------------------------------------------------------------------------
static constexpr int RERANK_K = 24;

// defined below; declared here so the re-ranker can read a reply's tool name
static std::string tool_name(const std::string& s);

// ---------------------------------------------------------------------------
//  THE REFUSAL TEXTS, in one place.
//
//  query_reranked can refuse a turn before retrieval runs, and returns -3 / -4 to
//  say so. Every caller must then produce the SAME message, because the difference
//  between "I do not understand you" and "I understand, and I do not do that" is
//  information the user acts on.
//
//  The text was written out twice -- once in am-chat, once in am_query_public --
//  and am-multi had a third behaviour: it mapped both codes to "(no match)",
//  reporting "understood but refused" as "nothing found". Three callers, three
//  messages, for one fact.
//
//  This is the same shape as defects 66 and 69 (a shared behaviour implemented at
//  individual call sites). The remedy is the same: state it once, here, next to the
//  code that produces the codes.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  THE CONFIDENCE FLOORS.
//
//  A request is acted on only when the nearest stored pattern is at least this
//  close. Two values because the situation differs:
//
//      structured  the sentence named a tool family, so wrong answers are rarer and
//                  the payload may be long or distinctive, which lowers the
//                  whole-sentence similarity for reasons unrelated to confidence.
//      unknown     only the embedding votes, so the bar is higher.
//
//  Measured separation on real text, which is what justifies specific numbers
//  rather than tuning them: in-scope requests score 0.532..1.000, out-of-scope
//  0.110..0.430.
//
//  Defined here rather than next to their original use because query_reranked --
//  earlier in the file, and the place that now ENFORCES them -- has to see them.
//  Moving the enforcement to the shared layer required moving the constants with
//  it, which is the honest consequence of that refactor: a guard and its threshold
//  belong together.
// ---------------------------------------------------------------------------
static constexpr float MIN_MATCH_STRUCTURED = 0.15f;   // family is known
static constexpr float MIN_MATCH_UNKNOWN    = 0.48f;   // only the embedding votes

const char* refusal_text(int code) {
  switch (code) {
    case -3:
      return "this request asks for two actions, which I do not do in one turn.\n"
             "  Please send them one at a time.";
    case -4:
      return "I understand the request but I have no tool for it.\n"
             "  I can: read a file, list a directory, write a file, run a shell\n"
             "  command, do arithmetic, report file metadata, search, make a\n"
             "  directory, or copy a file. Not available: delete, rename, move.";
    case -5:
      // Deliberately terse: -5 is raised when the nearest neighbour is too far away,
      // and the caller that has the similarity and the matched prompt can say more
      // than this layer can. A caller with nothing to add uses this text.
      return "I do not have a tool for that. The closest thing I know is not close\n"
             "  enough to act on, and acting on a guess is worse than asking you to\n"
             "  rephrase.";
    case -6:
      // The sentence pointed at a tool, but the thing to act on is not the kind of
      // thing that tool takes. Naming the disagreement is more useful than either a
      // bare refusal or a fabricated call.
      return "I cannot tell what to act on. The request names an operation, but the\n"
             "  part it would operate on is not the kind of thing that operation takes\n"
             "  (a path, an expression, or a command). Please rephrase.";
    default:
      return nullptr;
  }
}

// True for "the request was declined before any action", as opposed to "nothing
// found". -5 belongs here: it is a refusal, not an empty memory, and a caller that
// treats it as "no match" loses the distinction the whole refusal design exists for.
inline bool is_refusal(int code) {
  return code == -3 || code == -4 || code == -5 || code == -6;
}

static int query_reranked(AmMem& mem, const std::vector<float>& key,
                          const std::string& text,
                          std::vector<std::pair<float,int>>& top_out) {
  // ---------------------------------------------------------------------------
  //  MULTI-ACTION CHECK, AT THE SHARED LAYER.
  //
  //  This test lived in the am-chat turn handler, which meant every OTHER entry
  //  point lacked it. The automatic precedence hunt found that immediately: it
  //  calls am_query_public, which does not refuse two-action requests, so the
  //  combinations it built produced mixed tool calls that looked like precedence
  //  violations.
  //
  //  Six reported anomalies, all from this one gap. The lesson is the same shape
  //  as defect 51 and 57 (a rule placed at one call site rather than in the shared
  //  path): a behaviour that must apply to every caller belongs where every caller
  //  goes through.
  //
  //  am_query_public -> query_reranked is the common route, so the check is here.
  //  The caller in am-chat keeps its own, richer message; this one is the guard
  //  that makes the guarantee hold for every other entry point.
  // ---------------------------------------------------------------------------
  if (detect_multi_action(text)) {
    top_out.clear();
    return -3;   // distinct from -1 (empty memory): caller reports the refusal
  }

  // ---------------------------------------------------------------------------
  //  UNSUPPORTED CHECK, ALSO AT THE SHARED LAYER.
  //
  //  This is defect 66 happening a SECOND time, one function later: the
  //  multi-action guard was moved here, and the Unsupported guard -- which lived in
  //  the very next block of the same chat handler -- was left behind.
  //
  //  Measured consequence: the signal-injection harness reported
  //      "delete C:/tmp/x.txt"  ->  read_file
  //  while the same request through am-chat correctly refused. Two entry points,
  //  two behaviours, from one missing check.
  //
  //  Rather than remember to move the next guard too, the rule is stated: ANY
  //  decision that must hold for every caller lives in this function, and the
  //  callers only format the message. -3 and -4 are the two refusals this layer can
  //  produce; a caller that handles neither must at minimum not act on the result.
  // ---------------------------------------------------------------------------
  if (infer_intent(text) == Intent::Unsupported) {
    top_out.clear();
    return -4;   // understood, but no tool performs it
  }

  // ---------------------------------------------------------------------------
  //  CONFIDENCE FLOOR, AT THE SHARED LAYER. This is the THIRD time a guard has had
  //  to be moved here (defects 66 and 69 were the first two, both in the same
  //  evening), and this one is the most consequential of the three.
  //
  //  The floor lived in the am-chat turn handler. am_query_public -- the entry every
  //  TEST HARNESS uses -- only checked "bi < 0", so it acted on whatever the
  //  nearest neighbour happened to be, however distant.
  //
  //  Measured on CLINC150's out-of-scope split through am_query_public:
  //
  //      276 of 300 acted on (92.0%)
  //      "a show on broadway"                  -> a tool call
  //      "who has the best record in the nfl"  -> a tool call
  //      "can i mix antifreeze with water"     -> a tool call
  //
  //  while the SAME sentences through am-chat were refused correctly. Two entry
  //  points, opposite behaviours, from one missing check -- and because the harness
  //  used the permissive one, the system's own test suite had been measuring the
  //  wrong code path.
  //
  //  That last part is the general lesson: a harness that does not go through the
  //  same layer as production does not merely miss bugs, it HIDES them and reports
  //  success. Defects 67, 69 and 71 were harness bugs that produced false results;
  //  this is the mirror image -- a production bug that the harness was structurally
  //  unable to see.
  // ---------------------------------------------------------------------------
  const Intent want = infer_intent(text);

  // When the structure of the sentence names a tool family, search ONLY that
  // family. The encoder then has to discriminate between candidates that are
  // all already the right kind of thing -- a far easier problem than
  // discriminating across the whole 13500-row memory.
  //
  // The bucket array is indexed directly by Intent; an earlier version applied
  // a second remapping table on top of that and shifted every family by one,
  // which dropped overall accuracy to 0% while intent inference itself stayed
  // at 100%.
  // ---------------------------------------------------------------------------
  //  Bucketed search, with a confidence escape hatch.
  //
  //  THE CONFLICT THIS RESOLVES
  //  --------------------------
  //  Intent bucketing restricts the search to one tool family. That is what
  //  makes paraphrase work, but it also means a learned pattern can be
  //  UNREACHABLE: teaching "play some jazz" -> run_shell stored the row in the
  //  run_shell bucket, yet the query "play some jazz" infers read_file from the
  //  bare-path default, so the search never looked in the bucket that held the
  //  answer. The append succeeded, the delta file was written, and the answer was
  //  still wrong -- a silent failure that only showed up because the test asked
  //  for the taught response specifically.
  //
  //  The fix is not to abandon bucketing. It is to notice when the bucket is a
  //  bad guess: if the best match inside the inferred family is weak, the guess
  //  was probably wrong, so search the whole memory as well and take whichever
  //  answer is actually closer.
  //
  //  BUCKET_FLOOR is the similarity below which a family match is treated as
  //  unconvincing. Cosine between unrelated prompts measured ~0.02-0.15 in the
  //  am-sim probe, so 0.30 sits well above the noise floor.
  // ---------------------------------------------------------------------------
  static constexpr float BUCKET_FLOOR = 0.30f;
  std::vector<std::pair<float,int>> bucketed;
  int bi_bucket = -1;
  if (want != Intent::Unknown) {
    const std::vector<int>& rows = mem.bucket[AmMem::bucket_of(want)];
    if (!rows.empty()) {
      bi_bucket = mem.query_subset(key, rows, RERANK_K, bucketed);
    }
  }

  // ---------------------------------------------------------------------------
  //  THE ESCAPE HATCH, AND WHY IT IS GATED ON AN ABSOLUTE FLOOR.
  //
  //  The hatch exists for a good reason: intent bucketing can be WRONG, and when it
  //  is, the answer is in another family that this bucket will never surface. The
  //  taught-pattern case above is real.
  //
  //  But it fired on CORRECT buckets. Measured:
  //
  //      "fire up ipconfig"  intent=RunShell  (correct)
  //      run_shell bucket best = 0.1432
  //      global best           = 0.2628  (a search row)
  //      -> search {"path":"ipconfig"} -> refused -6, not a path
  //
  //  A first attempt gated the escape on a MARGIN (global must beat bucket by 0.10).
  //  0.12 > 0.10, so it still escaped. The margin was the wrong shape of test: it
  //  asks "is the other family clearly better than this one", when the real question
  //  is "is the other family any good at all".
  //
  //  Two weak numbers are not made decisive by their difference. 0.26 is a poor
  //  match on its own; that it beats 0.14 says only that both are poor. The escape
  //  now requires the global best to clear ESCAPE_FLOOR, an absolute bar set from
  //  the measured in-scope range (0.532..1.000 for structured requests). A candidate
  //  below it is not evidence of a wrong bucket, whatever it is ranked against.
  //
  //  The taught-pattern case still escapes: there the global answer scores high in
  //  absolute terms while the bucket scores ~0, which is exactly what this admits.
  // ---------------------------------------------------------------------------
  static constexpr float ESCAPE_FLOOR = 0.35f;  // Always compute the global best as well when the family match is weak.
  std::vector<std::pair<float,int>> global;
  const bool bucket_weak =
      (bi_bucket < 0) || bucketed.empty() || (bucketed[0].first < BUCKET_FLOOR);
  int bi_global = -1;
  if (bucket_weak) bi_global = mem.query(key, RERANK_K, global);

  if (getenv("AM_RANK_DIAG") && bi_bucket >= 0 && bi_global >= 0) {
    std::fprintf(stderr, "[rank] bucket best %.4f vs global best %.4f  (floor %.2f)\n",
                 bucketed.empty() ? -1.f : bucketed[0].first,
                 global.empty() ? -1.f : global[0].first, BUCKET_FLOOR);
  }
  // ---------------------------------------------------------------------------
  //  A SINGLE EXIT, SO THE FLOOR CANNOT BE BYPASSED.
  //
  //  The floor is applied HERE, after the best candidate has been chosen and its
  //  similarity is known. The first attempt placed the test before the search, where
  //  top_out is still empty, so the condition "top_out[0].first < floor" was never
  //  evaluated and the guard silently did nothing. Measured consequence: the
  //  out-of-scope rate stayed at 92% and "a show on broadway" was still routed to
  //  read_file {"path":"broadway"}.
  //
  //  That is the same shape as defects 51 and 57 -- a check placed where it cannot
  //  run -- and it is why the guard is now written as one exit path rather than
  //  repeated next to each return.
  //
  //  On a refusal the candidate list is KEPT: a caller measuring the boundary needs
  //  to know how far away the nearest neighbour was, and clearing it would erase the
  //  very quantity that justified declining.
  // ---------------------------------------------------------------------------
  int chosen = -1;
  if (bi_bucket >= 0 && bi_global >= 0) {
    const float sb = bucketed.empty() ? -1e9f : bucketed[0].first;
    const float sg = global.empty() ? -1e9f : global[0].first;
    // The global alternative must clear an ABSOLUTE floor to take over (see above).
    // A margin test was tried first and failed: two weak numbers, 0.14 vs 0.26, are
    // not made decisive by their difference.
    if (sg >= ESCAPE_FLOOR && sg > sb) { top_out = std::move(global); chosen = bi_global; }
    else                                { top_out = std::move(bucketed); chosen = bi_bucket; }
  } else if (bi_bucket >= 0) {
    top_out = std::move(bucketed);
    chosen = bi_bucket;
  } else if (bi_global >= 0) {
    top_out = std::move(global);
    chosen = bi_global;
  }
  if (chosen < 0) return -1;

  // ---------------------------------------------------------------------------
  //  AN EXPLICIT COMMAND TEMPLATE RE-PICKS THE CANDIDATE.
  //
  //  Measured: "run the command dir C:/tmp" was routed to list_dir, because the
  //  payload contains a path shape and a list_dir template scored higher.
  //  "run the command dir C:" survived only because that payload has no slash.
  //  The request said "run the command"; retrieving a directory listing from it is
  //  a retrieval error, not an ambiguity to be tolerated.
  //
  //  The first attempt at this fix set a local chosen_tool variable, which only
  //  changed what the VALIDATOR saw -- the executor still ran mem.replies[chosen].
  //  The check and the action disagreed, which is criterion 2 in miniature. The
  //  fix has to move `chosen` itself.
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  //  AN EXPLICIT COMMAND TEMPLATE RE-PICKS A run_shell CANDIDATE -- when there is one.
  //
  //  HISTORICAL NOTE, because this code was written under a wrong theory and the
  //  wrongness is the useful part.
  //
  //  Measured at the time: "run the command dir C:/tmp" produced 24 retrieval
  //  candidates that were ALL list_dir rows, so best_shell stayed -1. I read that
  //  as a REPRESENTATION limit -- "trigrams cannot tell dir the program from dir the
  //  directory name" -- and wrote it down as unfixable without a new encoder.
  //
  //  That was wrong. The candidate pool was not mis-ranked; it was never the right
  //  pool. infer_intent() had already returned ListDir, because the container-word
  //  rule ("dir ") sat ABOVE the command-template rule in the precedence table. The
  //  search then looked only in the ListDir bucket. By the time this function saw
  //  top_out, the run_shell rows had been filtered out several layers earlier.
  //
  //  Moving the template check to level 1.5 (above the container words, and without
  //  the old !path guard) took path-shaped commands from 0/3 to 3/3 with no change
  //  to the encoder at all.
  //
  //  The lesson: "the representation cannot express this" is a conclusion that needs
  //  the same evidence as any other, and a candidate pool that is uniformly wrong is
  //  a sign of an upstream filter, not of an inexpressive encoder. A homogeneous
  //  pool is the fingerprint of a bucket, not of a limit.
  //
  //  This block is now a safety net: it fires when a run_shell row IS present but
  //  ranked below a list_dir row.
  // ---------------------------------------------------------------------------
  if (!top_out.empty() && context::explicit_command_template(text)) {
    int best_shell = -1;
    float best_sim = -1e9f;
    for (const auto& cand : top_out) {
      const int idx = cand.second;
      if (idx < 0 || idx >= (int)mem.replies.size()) continue;
      if (AmMem::tool_name_of(mem.replies[idx]) != "run_shell") continue;
      if (cand.first > best_sim) { best_sim = cand.first; best_shell = idx; }
    }
    if (best_shell >= 0) chosen = best_shell;
  }


  {
    const bool structured = (want != Intent::Unknown);
    const float floor_v = structured ? MIN_MATCH_STRUCTURED : MIN_MATCH_UNKNOWN;
    if (!top_out.empty() && top_out[0].first < floor_v) return -5;
  }

  // ---------------------------------------------------------------------------
  //  THE PAYLOAD MUST BE THE KIND OF THING THE TOOL TAKES.
  //
  //  This is the context rule applied to the last step before acting: a tool that
  //  operates on paths cannot be given an expression, whatever the sentence looked
  //  like on the way in.
  //
  //  Measured, before this check:
  //      what is 7 + 8 in C:/tmp   ->  list_dir {"path":"7+8"}
  //  The sentence contains a path (C:/tmp), so the question+path rule claimed it as
  //  a listing; the payload extractor then produced the arithmetic, and nothing
  //  compared the two. The result names a directory that is neither the one
  //  mentioned nor a number.
  //
  //  Three separate rules now need this test -- the shell verb (defect 98), the
  //  calculator (92/93), and the path tools here -- which is why it lives at the
  //  shared layer instead of being restated per rule.
  //
  //  The test is deliberately narrow: it rejects a payload containing characters
  //  that cannot appear in a path segment and accepts everything else. A bare
  //  relative name like "build" is still accepted, because "make a directory build"
  //  is a legitimate request and this check is not meant to police spelling.
  // ---------------------------------------------------------------------------
  {
    const std::string pay = extract_payload(text);
    if (pay.empty()) return chosen;

    // -------------------------------------------------------------------------
    //  AN EXACT MATCH IS AUTHORITATIVE -- the payload was STORED, not fabricated.
    //
    //  This check exists to catch payloads invented from a weak match: the extractor
    //  falls back to "the last whitespace token", so an unrelated sentence yields an
    //  argument that nothing in the request supports. That reasoning does not apply
    //  when the memory contains this exact prompt, because then the payload came from
    //  the store.
    //
    //  Measured regression without this guard: am-learntest fell from 4/4 to 0/4.
    //  A user who teaches "play some jazz" -> run_shell {"cmd":"play jazz"} has
    //  stated the mapping explicitly, and the router answered "I cannot tell what to
    //  act on" -- applying a heuristic against the one source of truth that outranks
    //  heuristics. Learning exists precisely so that a taught pattern beats a guess.
    //
    //  THE THRESHOLD IS 0.70, NOT 0.95, and the difference is the point.
    //
    //  At 0.95 only the verbatim taught sentence passed, so am-learntest recovered to
    //  1/4 -- the three TEMPLATE cases still failed. Those are the cases the system
    //  exists to serve: "play some blues" must reuse the taught "play some jazz"
    //  pattern and swap the payload. Blocking them protects nothing and removes a
    //  designed capability.
    //
    //  The distinction the threshold encodes is confidence about the STORED PATTERN,
    //  not about the payload. At 0.70 the stored prompt is nearly this sentence, so
    //  its tool choice is grounded and the payload is the variable part -- slot
    //  transfer is exactly the intended mechanism. Below 0.70 the match is a guess,
    //  and a guessed tool plus a fallback-extracted payload is how "tuning notes"
    //  became a file read.
    // -------------------------------------------------------------------------
    //  THE EXEMPTION IS NARROWER THAN IT LOOKS, and mutation testing showed why.
    //
    //  It bypassed this whole block, including the TYPE test below. So
    //      "read the file build"   ->  read_file {"path":"build"}
    //  -- a bare English word accepted as a path -- because the sentence matched the
    //  stored "read the file <X>" template at 0.76 and the exemption returned early.
    //
    //  The exemption exists so a TAUGHT pattern beats a heuristic about whether the
    //  payload was invented. It says nothing about whether the payload is the right
    //  KIND of thing, and those are separate questions: a confidently retrieved
    //  template can still be handed a payload the tool cannot use.
    //
    //  So it now skips only the fabrication check, not the type check.
    const bool confident = (!top_out.empty() && top_out[0].first >= 0.70f);

    // -------------------------------------------------------------------------
    //  THE TOOL THAT WILL ACTUALLY RUN, not the intent that was inferred.
    //
    //  The first version keyed on the inferred intent, and that made it miss the
    //  cases it was written for: when infer_intent returns Unknown, no tool is
    //  named, the check is skipped, and retrieval then picks a candidate whose reply
    //  DOES call one. Measured:
    //
    //      "read text"                     -> read_file {"path":"text"}
    //      "the most current market news"  -> read_file {"path":"news"}
    //
    //  In both, the intent was Unknown and the tool came from the nearest neighbour,
    //  so the payload was never examined. Keying on the chosen reply closes that gap
    //  and states the invariant more honestly: the payload must fit the tool that is
    //  about to run, whatever led to it.
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    //  AN EXPLICIT COMMAND TEMPLATE LOCKS THE TOOL, not just the payload check.
    //
    //  Measured: "run the command dir C:/tmp" was routed to list_dir, because the
    //  payload contains a path shape and retrieval found a list_dir template
    //  closer. The payload gate then rejected it with -6 and the user got a
    //  refusal for a perfectly ordinary command. "run the command dir C:" did the
    //  same thing only because that payload has no path shape.
    //
    //  The sentence said "run the command". That is not a hint to be weighed
    //  against cosine similarity -- it is a statement about which tool applies. A
    //  request that names its own shape should not be overruled by a neighbour
    //  that matches its payload better.
    // -------------------------------------------------------------------------
    std::string chosen_tool = (chosen >= 0 && chosen < (int)mem.replies.size())
        ? AmMem::tool_name_of(mem.replies[chosen]) : std::string();
    bool tool_locked = false;
    if (!confident && context::explicit_command_template(text)) {
      chosen_tool = "run_shell";
      tool_locked = true;
    }

    //  Reading, listing and stat-ing need a FILE REFERENCE: a bare word is not one.
    //
    //  Measured on CLINC150 out-of-scope sentences, every remaining false action was
    //  a fabricated payload produced by the "last whitespace token" fallback:
    //
    //      "can you give me the standard tuning notes"  -> read_file {"path":"notes"}
    //      "read text"                                  -> read_file {"path":"text"}
    //      "the most current market news"               -> read_file {"path":"news"}
    //
    //  In each, the payload is the last word of a sentence about the world. Requiring
    //  path shape rejects all three while leaving "STATUS.md" and "C:/tmp/x.txt"
    //  untouched.
    //
    //  mkdir and write_file are deliberately excluded: there, NAMING something is the
    //  point, so "make a directory build" is a legitimate bare name.
    //  search and copy take paths too. The first version of this list held only
    //  read_file/list_dir/stat, so "open the file config" was routed to
    //      search {"path":"config","pattern":...}
    //  with a bare word as the directory -- the same defect, through a tool the check
    //  did not name. Listing the tools that consume a path is exactly the kind of
    //  hand-kept set this project keeps getting burned by; the honest version is
    //  "every tool whose payload is a path", which is all of them but calc and
    //  run_shell.
    const bool takes_path = (chosen_tool == "read_file" || chosen_tool == "list_dir" ||
                             chosen_tool == "stat"      || chosen_tool == "search"   ||
                             chosen_tool == "copy");
    // NOT exempted by confidence: whether a payload is a file reference is a fact
    // about the payload, and a confident template match does not make "build" a path.
    if (takes_path && !context::licensed_as_file_reference(pay)) return -6;

    //  A shell command needs VOCABULARY, not just ASCII.
    //
    //      "…starts running low"                -> run_shell {"cmd":"low"}
    //      "…state college run"                 -> run_shell {"cmd":"run"}
    //
    //  "low" and "run" are single ASCII words, exactly like "hostname" -- no shape
    //  test separates them. What separates them is that "hostname" is the name of a
    //  program. The vocabulary lives in context.h, and a multi-token span or a path
    //  to an executable carries its own evidence.
    //
    //  AND THE CHECK APPLIES ONLY TO PROSE-LENGTH SENTENCES.
    //
    //  Requiring the command vocabulary unconditionally fixed two fabrications and
    //  broke two designed capabilities (am-learntest fell to 2/4): "play some opera"
    //  must reuse the taught "play some jazz" pattern, and its payload "opera" is not
    //  a command name either.
    //
    //  The difference between the two situations is the SENTENCE, not the payload:
    //
    //      "…starts running low"   10 tokens -- the payload is one word taken out of
    //                              prose that is not asking to run anything
    //      "play some opera"        3 tokens -- the payload IS the request
    //
    //  So the vocabulary requirement is applied when the sentence is long enough to
    //  be prose. Six tokens is where the measured examples separate: every false
    //  action came from a sentence of eight or more words, and every legitimate
    //  template transfer from one of four or fewer.
    //
    //  This is a heuristic and is labelled as one. What makes it acceptable is the
    //  asymmetry it protects: acting on "low" runs a program that does not exist,
    //  while declining "play some opera" costs the user one rephrase.
    if (chosen_tool == "run_shell") {
      int tokens = 0;
      bool in_tok = false;
      for (unsigned char c : text) {
        const bool sp = (c == ' ' || c == '\t' || c == ',' || c == '.' || c == '?');
        if (!sp && !in_tok) { tokens++; in_tok = true; }
        else if (sp) in_tok = false;
      }
      // The confidence exemption applies HERE: a taught mapping such as
      // "play some jazz" -> run_shell {"cmd":"play jazz"} is authoritative, and its
      // payload is not a vocabulary word. This is the fabrication question, so a
      // confident match settles it.
      // TWO PATHS, and the distinction is the TEMPLATE, not the length.
      //
      //  Explicit template ("run the command X", "execute X"): the sentence has
      //  already asserted that X is a program, so the vocabulary must confirm it.
      //  Measured gap this closes: "run the command zzznotacommand" is one token
      //  after the verb, so the old length test skipped the check entirely and an
      //  invented program name was executed.
      //
      //  Vague lead ("run X", "call X", "start X"): length still stands in for
      //  "prose with a stray last word", because "start the meeting notes" and
      //  "start notepad" have the same shape and only the vocabulary separates them.
      if (!confident) {
        const bool explicit_tpl = context::explicit_command_template(text);
        const bool licensed = context::licensed_as_command_strong(pay);
        if (explicit_tpl && !licensed) return -6;
        if (!explicit_tpl && tokens > 6 && !licensed) return -6;
      }
    }
  }
  return chosen;
}

static std::string tool_name(const std::string& s) {
  size_t a = s.find("<|tool|>");
  if (a == std::string::npos) return "";
  size_t b = s.find(' ', a);
  return s.substr(a + 8, (b == std::string::npos ? s.size() : b) - (a + 8));
}

} // namespace canticle

// ============================================================================
//  CLI
// ============================================================================
using namespace canticle;

int am_run(int argc, char** argv) {
  std::string mode = argc > 0 ? argv[0] : "";
  std::string ampath = "build\\ckpt\\canticle.am";
  std::string prompt;
  int n_samples = 6000, seed = 2026, K = 3;
  bool diag = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--am"     && i+1 < argc) ampath = argv[++i];
    if (a == "--prompt" && i+1 < argc) prompt = argv[++i];
    if (a == "--data"   && i+1 < argc) n_samples = atoi(argv[++i]);
    if (a == "--topk"   && i+1 < argc) K = atoi(argv[++i]);
    if (a == "--beta"   && i+1 < argc) AM_BETA = (float)atof(argv[++i]);
    if (a == "--steps"  && i+1 < argc) AM_STEPS = atoi(argv[++i]);
    if (a == "--diag")  diag = true;
  }
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--wlex" && i + 1 < argc) HYB_W_LEX = (float)atof(argv[++i]);
  std::printf("[cfg] D=%d beta=%.1f cleanup=%d topk=%d  w_gram=%.2f w_lex=%.2f\n",
              AM_D, AM_BETA, AM_STEPS, K, HYB_W_GRAM, HYB_W_LEX);

  auto samples = synth_tool_data(n_samples, seed);
  std::vector<std::string> docs;
  for (auto& s : samples) docs.push_back(s.prompt);
  AmEncoder enc; enc.D = HYB_GRAM_DIM;
  LexModel lex; lex.DIM = HYB_LEX_DIM;
  // Train the lexical model on the generator corpus now; the same corpus the
  // memory will be built from, so co-occurrence is in-distribution.
  {
    auto tr = synth_tool_data(4000, seed);
    std::vector<std::string> corpus;
    for (auto& s : tr) corpus.push_back(s.prompt);
    lex.train(corpus, HYB_LEX_DIM);
    std::printf("[lex] trained on %zu sentences, vocab %zu\n", corpus.size(), lex.words.size());
  }
  auto encode_hybrid = [&](const std::string& text, std::vector<float>& out) {
    std::vector<float> g, l;
    enc.encode(text, g);
    lex.encode(text, l);
    hybrid_key(g, l, out);
  };
  double t0 = now_ms();
  enc.build_idf(docs);
  std::printf("[idf] built over %zu prompts in %.0f ms\n", docs.size(), now_ms() - t0);

  if (mode == "am-build") {
    AmMem mem;
    double t1 = now_ms();
    std::vector<std::vector<float>> keys; std::vector<std::string> ps, rs;
    keys.reserve(samples.size());
    for (auto& s : samples) {
      std::vector<float> v; encode_hybrid(s.prompt, v);
      keys.push_back(std::move(v)); ps.push_back(s.prompt); rs.push_back(s.reply);
    }
    mem.store_batch(keys, ps, rs);
    double t2 = now_ms();
    std::printf("stored %d patterns in %.1f ms (%.1f us/pattern, 1 pass, 0 gradients)\n",
                mem.N, t2 - t1, (t2 - t1) * 1000.0 / std::max(1, mem.N));
    std::filesystem::path p(ampath);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    if (mem.save(ampath)) std::printf("[saved] %s\n", ampath.c_str());
    return 0;
  }

  if (mode == "am-eval") {
    const size_t ntr = samples.size() * 4 / 5;
    AmMem mem;
    std::vector<std::vector<float>> keys; std::vector<std::string> ps, rs;
    keys.reserve(ntr);
    for (size_t i = 0; i < ntr; ++i) {
      std::vector<float> v; encode_hybrid(samples[i].prompt, v);
      keys.push_back(std::move(v)); ps.push_back(samples[i].prompt); rs.push_back(samples[i].reply);
    }
    mem.store_batch(keys, ps, rs);
    std::vector<float> v;
    std::printf("=== CANTICLE-AM generalisation ===\n");
    std::printf("stored %d patterns (ONE pass, no gradients, no epochs)\n", mem.N);

    bool norank = false;
    for (int i = 1; i < argc; ++i)
      if (std::string(argv[i]) == "--norank") norank = true;
    int tool_ok = 0, tool_tot = 0, exact = 0, shown = 0, top1_right = 0;
    for (size_t i = ntr; i < samples.size(); ++i) {
      encode_hybrid(samples[i].prompt, v);
      std::vector<std::pair<float,int>> top;
      // Use the SAME retrieval the chat path uses. This line previously called
      // mem.query() directly, skipping intent bucketing; with hybrid keys the
      // lexical part then pulled "what is 54 + 31" toward the stored
      // "what is in C:/..." listing prompts, which share the words "what"/"is",
      // and three arithmetic requests were routed to list_dir.
      // Evaluation must exercise the production path or it measures nothing.
      // Ablation switch: --norank bypasses intent bucketing so the two retrieval
      // paths can be compared on identical data.
      int bi = norank ? mem.query(v, K, top)
                      : query_reranked(mem, v, samples[i].prompt, top);
      // -3 and -4 are refusals produced before retrieval. Scoring them as correct
      // would be wrong (no tool was chosen) and scoring them as a wrong tool would
      // also be wrong (nothing was chosen). They are counted as "no call", which is
      // what the tool-name comparison naturally reports for a refusal string.
      if (bi < 0) {
        if (is_refusal(bi)) { tool_tot++; }
        continue;
      }
      std::string got = slot_transfer(samples[i].prompt, mem.prompts[bi], mem.replies[bi]);
      tool_tot++;
      if (tool_name(got) == tool_name(samples[i].reply)) tool_ok++;
      if (got == samples[i].reply) exact++;
      const bool exact_hit = (got == samples[i].reply);
      if (diag && !exact_hit && shown < 25) {
        shown++;
        std::printf("\n  MISMATCH\n    in        : %s\n    want      : %s\n    got       : %s\n"
                    "    matched   : %s\n    q_payload : '%s'\n    m_payload : '%s'\n",
                    samples[i].prompt.c_str(), samples[i].reply.c_str(), got.c_str(),
                    mem.prompts[top[0].second].c_str(),
                    extract_payload(samples[i].prompt).c_str(),
                    extract_payload(mem.prompts[top[0].second]).c_str());
      } else if (!diag && shown < 6) {
        shown++;
        std::printf("\n  in   : %s\n  want : %s\n  got  : %s\n  [%s] sim=%.3f\n",
                    samples[i].prompt.c_str(), samples[i].reply.c_str(), got.c_str(),
                    exact_hit ? "EXACT" : "differs", top[0].first);
      }
    }
    if (tool_tot) {
      std::printf("\n--- held-out (unseen prompts) ---\n");
      std::printf("  tool-name accuracy : %.2f%%  (%d/%d)\n", 100.0*tool_ok/tool_tot, tool_ok, tool_tot);
      std::printf("  exact reply match  : %.2f%%  (%d/%d)\n", 100.0*exact/tool_tot, exact, tool_tot);
    }
    return 0;
  }

  if (mode == "am-ask") {
    AmMem mem;
    if (!mem.load_with_delta(ampath, enc, lex)) { std::fprintf(stderr, "cannot load %s\n", ampath.c_str()); return 2; }
    std::vector<float> v;
    encode_hybrid(prompt, v);
    std::vector<std::pair<float,int>> top;
    // ---------------------------------------------------------------------------
    //  am-ask GOES THROUGH THE SHARED LAYER, like every other entry point.
    //
    //  It called mem.query directly, so it had NO multi-action check, NO Unsupported
    //  check, NO confidence floor and NO payload-kind check. Measured by an
    //  independent audit:
    //
    //      am-ask --prompt "delete C:/tmp/x.txt"          -> read_file {"path":...}
    //      am-ask --prompt "read C:/a.txt and list C:/tmp" -> list_dir {"path":"C:/a.txt"}
    //
    //  -- the first is defect 69 back from the dead, the second is defect 33's
    //  clause mixing, both through a command registered in the user-facing command
    //  map. So the claim that moving the guards to query_reranked made every entry
    //  point safe was false: am-ask was simply not one of the paths anyone checked.
    //
    //  The candidate list below is still printed, because seeing the neighbours is
    //  the reason this command exists. What changes is that the ANSWER now obeys the
    //  same rules as a real turn -- otherwise this command answers questions the
    //  system would refuse to answer anywhere else.
    // ---------------------------------------------------------------------------
    const int bi = query_reranked(mem, v, prompt, top);
    if (const char* rf = refusal_text(bi)) {
      std::printf("user> %s\ncanticle> %s\n", prompt.c_str(), rf);
      std::printf("[refused before acting, code %d -- gates applied]\n", bi);
      return 0;
    }
    if (bi < 0) { std::printf("empty memory\n"); return 1; }
    std::string got = slot_transfer(prompt, mem.prompts[bi], mem.replies[bi]);
    std::printf("user> %s\ncanticle> %s\n", prompt.c_str(), got.c_str());
    std::printf("[%zu candidates, cosine %.3f..%.3f]\n", top.size(),
                top.back().first, top.front().first);
    for (auto& t : top) std::printf("    %.3f  %s\n", t.first, mem.prompts[t.second].c_str());
    return 0;
  }

  if (mode == "am-learn") {
    // -----------------------------------------------------------------------
    //  Online learning: --teach  adds a pattern,  --correct  fixes a mistake.
    //  Both take effect immediately and persist to the delta file, without a
    //  rebuild of the base memory.
    // -----------------------------------------------------------------------
    std::string teach, teach_reply, correct_q, correct_r;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--teach"   && i + 2 < argc) { teach = argv[i+1]; teach_reply = argv[i+2]; }
      if (a == "--correct" && i + 2 < argc) { correct_q = argv[i+1]; correct_r = argv[i+2]; }
      // A reply containing JSON quotes is painful to pass through a shell, and
      // an earlier attempt silently stored a truncated reply because the shell
      // ate the quotes. Reading the reply from a file removes the escaping layer
      // entirely -- the bytes on disk are the bytes stored.
      if (a == "--teach-file" && i + 2 < argc) {
        teach = argv[i+1];
        std::ifstream f(argv[i+2], std::ios::binary);
        if (!f) { std::fprintf(stderr, "[learn] cannot read %s\n", argv[i+2]); return 4; }
        std::stringstream ss; ss << f.rdbuf();
        teach_reply = ss.str();
        while (!teach_reply.empty() &&
               (teach_reply.back() == '\n' || teach_reply.back() == '\r'))
          teach_reply.pop_back();
      }
    }
    if (teach.empty() && correct_q.empty()) {
      std::fprintf(stderr,
        "usage: canticle am-learn [--am FILE] --teach \"PROMPT\" \"REPLY\"\n"
        "       canticle am-learn [--am FILE] --correct \"PROMPT\" \"REPLY\"\n");
      return 1;
    }

    AmMem mem;
    if (!mem.load_with_delta(ampath, enc, lex)) {
      std::fprintf(stderr, "cannot load %s\n", ampath.c_str());
      return 2;
    }
    const int before = mem.N;
    const int replayed = mem.replay_delta(ampath, enc, lex);
    if (replayed) std::printf("[delta] replayed %d previously learned patterns\n", replayed);

    auto do_add = [&](const std::string& p, const std::string& r) -> bool {
      std::vector<float> g, l, key;
      enc.encode(p, g);
      lex.encode(p, l);
      hybrid_key(g, l, key);
      std::string err;
      if (!mem.append(key, p, r, &err)) {
        std::fprintf(stderr, "[learn] append failed: %s\n", err.c_str());
        return false;
      }
      if (!mem.append_delta(ampath, p, r)) {
        std::fprintf(stderr, "[learn] could not persist to delta\n");
        return false;
      }
      return true;
    };

    double t0 = now_ms();
    if (!teach.empty()) {
      if (!do_add(teach, teach_reply)) return 3;
      std::printf("[learn] taught: %s -> %s\n", teach.c_str(), teach_reply.c_str());
    }
    // Query through THIS mem instance, not through am_query_public: the latter
    // keeps its own static AmMem and never replays the delta, so it would report
    // a stale answer and the before/after comparison would be meaningless.
    auto ask = [&](const std::string& q) -> std::string {
      std::vector<float> g, l, key;
      enc.encode(q, g);
      lex.encode(q, l);
      hybrid_key(g, l, key);
      std::vector<std::pair<float,int>> top;
      const int bi = query_reranked(mem, key, q, top);
      // A refusal is not "no match". The earlier version reported both as
      // "(no match)", which hides the difference between "understood but declined"
      // and "nothing found" -- the distinction this system exists to preserve.
      if (const char* rf = refusal_text(bi)) return rf;
      if (bi < 0) return "(no match)";
      return slot_transfer(q, mem.prompts[bi], mem.replies[bi]);
    };

    if (!correct_q.empty()) {
      // Record what the answer actually was before overwriting, so the log shows
      // the mistake rather than assuming one.
      const std::string was = ask(correct_q);
      std::printf("[learn] before correction: %s\n", was.c_str());
      if (!do_add(correct_q, correct_r)) return 3;
      const std::string now = ask(correct_q);
      std::printf("[learn] after  correction: %s\n", now.c_str());
      std::printf("[learn] %s\n", (now == correct_r) ? "CORRECTED" : "STILL DIFFERS");
    }
    const double ms = now_ms() - t0;
    std::printf("[learn] memory %d -> %d patterns in %.2f ms (%.0f us/pattern, no rebuild)\n",
                before, mem.N, ms, ms * 1000.0 / std::max(1, mem.N - before));
    return 0;
  }

  if (mode == "am-harvest") {
    // -----------------------------------------------------------------------
    //  Review and apply the teaching queue.
    //
    //  Default is DRY RUN: it prints what would be taught and why. --apply
    //  commits. A learning system that writes to persistent memory without a
    //  review step is one bad observation away from permanent corruption, and
    //  the admission rule alone is not strong enough evidence to skip review.
    // -----------------------------------------------------------------------
    bool apply = false;
    bool no_collapse = false;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--apply") apply = true;
      // Ablation switch: store every example as its own row. Needed to measure
      // what collapsing actually costs -- an earlier comparison called harvest on
      // both variants and therefore collapsed both, making the two runs identical
      // by construction.
      if (std::string(argv[i]) == "--no-collapse") no_collapse = true;
    }

    LearnQueue lq(ampath + ".queue");
    lq.evaluate();
    const auto all = lq.all();
    const auto acc = lq.accepted();

    std::printf("=== teaching queue: %zu observation(s) ===\n\n", all.size());
    for (const auto& o : all) {
      std::printf("  [%s] hits=%d  sim=%.2f  %s\n",
                  o.accepted ? "ADMIT" : "  -  ", o.hits, o.top_sim, o.reason);
      std::printf("        %s\n", o.query.c_str());
    }
    std::printf("\n  %zu admitted of %zu  (need hits>=%d and sim<%.2f)\n",
                acc.size(), all.size(), MIN_SUPPORT, SIM_CEILING);

    // ---- template induction over the admitted set ----
    //
    // Storing one row per observation scales with how talkative users are. If
    // several admitted observations share a surface frame, one canonical row with
    // a payload slot replaces them all. Printed here rather than applied silently:
    // a frame is a CLAIM about structure, and a wrong claim is worse than a
    // missing row because it will match requests it should not.
    //
    // Induction runs over ALL observations, not only the admitted ones.
    //
    // Why: a corrected pattern starts working IMMEDIATELY -- retrieval generalises
    // through slot transfer, so the second mention of the same frame no longer
    // fails and never becomes a correction. Measured on
    // "play some jazz"(fail) -> "run the command jazz"(success), the follow-up
    // "play some blues" already succeeded on its own, so it was recorded as an
    // ordinary observation with no failure to pair against.
    //
    // Restricting induction to the admitted set therefore missed exactly the
    // repeated frames it was built to find. The observations carry the structure;
    // admission only decides what gets written as an individual row.
    {
      // Failed observations are INCLUDED with an empty reply: their query text is
      // what defines the frame, and induce_frames takes the action template from
      // whichever member has one.
      std::vector<std::pair<std::string,std::string>> obs;
      for (const auto& o : all) obs.push_back({o.query, o.reply});
      const auto frames = induce_frames(obs);
      if (!frames.empty()) {
        std::printf("\n  --- induced frames (%zu) ---\n", frames.size());
        for (const auto& f : frames) {
          std::printf("    \"%s\"  from %d observation(s), %zu distinct payload(s)\n",
                      f.text.c_str(), f.observations, f.payloads.size());
          std::printf("        action: %s\n", f.action_template.c_str());
        }
      } else if (obs.size() >= MIN_SUPPORT_FOR_FRAME) {
        std::printf("\n  no frame: %zu observation(s) share no structure worth storing\n",
                    obs.size());
      }
    }

    // ---- collapse frame members into ONE row per family ----
    //
    // The claim under test: three examples of "play some X" do not need three
    // memory rows. Retrieval already generalises across a family through slot
    // transfer, so the row count can track the number of RULES rather than the
    // number of things people say.
    //
    // The frame's canonical row is the frame text itself with a literal payload,
    // because that is what the encoder will see at query time: a user request
    // contains characters, not a placeholder. Any member's payload works as the
    // representative; the first is used.
    std::vector<std::pair<std::string,int>> collapse;   // (query, frame index)
    std::vector<Frame> frames_cache;
    {
      std::vector<std::pair<std::string,std::string>> obs;
      for (const auto& o : all) obs.push_back({o.query, o.reply});
      const auto frames = induce_frames(obs);
      frames_cache = frames;
      if (!frames.empty()) {
        std::printf("\n  --- frames declared for writing ---\n");
        for (const auto& f : frames)
          std::printf("    \"%s\" -> %s\n", f.text.c_str(), f.action_template.c_str());
      }
      // Only REMOVE examples that are durable members of a frame; the frame row
      // replaces them. A frame needs at least MIN_SUPPORT_FOR_FRAME payloads,
      // which induction already required.
      for (int fi = 0; fi < (int)frames.size(); ++fi)
        for (const auto& o : acc)
          if (frame_matches(frames[fi], o.query)) collapse.push_back({o.query, fi});
    }

    if (!apply) {
      std::printf("\n  dry run. re-run with --apply to commit.\n");
      return 0;
    }
    // NOTE: this used to return early when no individual observation was
    // admitted. That made frames unreachable: three DIFFERENT requests like
    // "play some jazz/blues/metal" each have hits == 1 and none is admitted, yet
    // they are exactly what induction is for, and a frame admits itself. The
    // guard now also requires that no frame was found.
    if (acc.empty() && frames_cache.empty()) {
      std::printf("\n  nothing to apply.\n");
      return 0;
    }

    AmMem mem;
    if (!mem.load_with_delta(ampath, enc, lex)) {
      std::fprintf(stderr, "cannot load %s\n", ampath.c_str());
      return 2;
    }
    // ---- which frame members collapse, and who represents each frame ----
    //
    // The representative is the FIRST member of the frame: any member works,
    // because retrieval substitutes the payload at query time.
    // -----------------------------------------------------------------------
    //  EVERY admitted example covered by a frame is suppressed, INCLUDING the one
    //  whose payload the frame row happens to reuse.
    //
    //  The first version kept the "representative" member and suppressed only the
    //  others. Measured result: with one admitted example the collapsed run wrote
    //  the frame row AND that same example again -- 3377 rows against the
    //  uncollapsed 3376. Collapsing made the memory LARGER, which is the opposite
    //  of the point, and the extra row was a duplicate of the frame's own content.
    //
    //  The representative is therefore an implementation detail of how the frame
    //  row gets a concrete query string, not a member that survives alongside it.
    // -----------------------------------------------------------------------
    std::map<int, std::string> representative;
    std::set<std::string> suppressed;
    if (!no_collapse) {
      for (const auto& c : collapse) {
        if (representative.find(c.second) == representative.end())
          representative[c.second] = c.first;   // payload source for the frame row
        suppressed.insert(c.first);             // and the example itself is covered
      }
    }

    // -----------------------------------------------------------------------
    //  A FRAME ADMITS ITSELF.
    //
    //  Measured problem: with three DIFFERENT observations like "play some jazz",
    //  "play some blues" and "play some metal", every entry has hits == 1, so the
    //  support test admits none of them -- and there was therefore nothing for
    //  collapsing to collapse. The feature was unreachable on exactly the input it
    //  exists for.
    //
    //  The support rule is the wrong test for a frame. It was written to stop a
    //  single example masquerading as a rule, but a frame is supported by
    //  MIN_SUPPORT_FOR_FRAME DISTINCT payloads, which is strictly stronger
    //  evidence than seeing one request twice: the same rule has been observed
    //  across three separate things people said.
    //
    //  So a frame is written on its own authority, using an explicit member as the
    //  representative row. Its members are then suppressed, and neither needs to
    //  satisfy the per-observation support test.
    // -----------------------------------------------------------------------
    int added = 0, collapsed = 0;
    const double t0 = now_ms();

    // 1) the frames themselves
    for (const auto& kv : representative) {
      const int fi = kv.first;
      if (fi < 0 || fi >= (int)frames_cache.size()) continue;
      const Frame& f = frames_cache[fi];
      // Query text: the frame with its first payload filled in, so the stored row
      // is a real sentence the encoder can compare against.
      std::string q = f.text;
      const size_t brace = q.find("{}");
      if (brace != std::string::npos && !f.payloads.empty())
        q = q.substr(0, brace) + f.payloads[0] + q.substr(brace + 2);
      std::string a = f.action_template;
      const size_t ab = a.find("{}");
      if (ab != std::string::npos && !f.payloads.empty())
        a = a.substr(0, ab) + f.payloads[0] + a.substr(ab + 2);

      std::vector<float> g, l, key;
      enc.encode(q, g);
      lex.encode(q, l);
      hybrid_key(g, l, key);
      if (mem.append(key, q, a) && mem.append_delta(ampath, q, a)) added++;
    }
    const int frame_rows = added;

    // 2) admitted observations that no frame covers
    for (const auto& o : acc) {
      if (suppressed.count(o.query)) { collapsed++; continue; }
      std::vector<float> g, l, key;
      enc.encode(o.query, g);
      lex.encode(o.query, l);
      hybrid_key(g, l, key);
      if (mem.append(key, o.query, o.reply) && mem.append_delta(ampath, o.query, o.reply))
        added++;
    }
    const double ms = now_ms() - t0;
    std::printf("\n  applied %d pattern(s) in %.2f ms (%.0f us each)\n",
                added, ms, ms * 1000.0 / std::max(1, added));
    if (frame_rows)
      std::printf("  of which %d are frame rows representing %zu example(s)\n",
                  frame_rows, collapse.size());

    // Applied entries are removed from the queue, otherwise every harvest would
    // re-append them and the memory would grow without bound.
    for (const auto& o : acc) lq.items.erase(LearnQueue::norm(o.query));
    lq.save();
    std::printf("  queue now holds %zu observation(s)\n", lq.items.size());
    return 0;
  }

  if (mode == "am-chat" || mode == "am-bench") {
    AmMem mem;
    // Must replay the delta: this is the interactive path, and a learned pattern
    // is invisible without it. An earlier version used plain load() here and
    // every taught answer silently reverted on the next process start.
    if (!mem.load_with_delta(ampath, enc, lex)) {
      std::fprintf(stderr, "cannot load %s\n", ampath.c_str());
      return 2;
    }
    // Keys are stored fp16. The first version of this line reported N*D*4,
    // i.e. the fp32 size, so every bandwidth figure printed below was inflated
    // by exactly 2x. Measured against the real allocation the number is halved
    // and, importantly, no longer exceeds the DRAM copy ceiling.
    if (getenv("AM_STATS_ONLY")) {
      std::printf("STATS episodes=%d dims=%d\n", mem.N, mem.D);
      return 0;
    }
    std::printf("CANTICLE-AM  %d episodes x %d dims  (%.1f MB of fp16 keys)\n",
                mem.N, mem.D, mem.N * (double)mem.D * 2 / 1048576.0);

    // ---- latency benchmark: the number that matters for "high speed" ----
    if (mode == "am-bench") {
      // build a few encoded queries (each row is already part of the test set,
      // which is fine for a latency measurement)
      std::vector<std::vector<float>> qs;
      std::vector<std::pair<float,int>> top;
      for (size_t i = mem.prompts.size() * 4 / 5; i < mem.prompts.size() && qs.size() < 256; ++i) {
        std::vector<float> v; encode_hybrid(mem.prompts[i], v); qs.push_back(std::move(v));
      }
      bool use_f16 = false;
      for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--f16") use_f16 = true;
      std::printf("  mode              : %s\n",
                  use_f16 ? "fp16 keys (baseline)" : "ternary keys (2 bit/dim, popcount)");
      const int REP = 1000;
      std::vector<int> sanity;
      for (int r = 0; r < REP; ++r) {
        int bi = use_f16 ? mem.query(qs[r % qs.size()], 1, top)
                         : mem.query_ternary(qs[r % qs.size()], 1, top);
        if (r < 8) sanity.push_back(bi);
      }
      double t1 = now_ms();
      (void)sanity;
      const double per = (t1 - t0) / REP;
      std::printf("\n--- retrieval latency ---\n");
      std::printf("  episodes          : %d\n", mem.N);
      const double kbytes = use_f16 ? mem.N * (double)mem.D * 2
                                    : mem.N * (double)mem.D * 0.25;   // 2 bits/dim
      std::printf("  key matrix        : %.2f MB (%s)\n", kbytes / 1048576.0,
                  use_f16 ? "fp16" : "ternary, 2 bit/dim");
      std::printf("  per query         : %.3f ms  ->  %.0f queries/s\n", per, 1000.0 / per);
      std::printf("  bytes read/query  : %.1f MB\n", kbytes / 1048576.0);
      std::printf("  effective BW      : %.1f GB/s\n", kbytes / 1e9 / (per / 1000.0));
      return 0;
    }

    // ---- interactive / scripted conversation with real tool execution ----
    Session sess;
    bool dbg = false;
    bool learn_on = false;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--debug") dbg = true;
      if (std::string(argv[i]) == "--learn") learn_on = true;
    }
    // Autonomous learning is opt-in: it writes to disk, and a listener that
    // silently accumulates observations would be a surprise.
    LearnQueue lq(ampath + ".queue");
    LearnQueue* queue = learn_on ? &lq : nullptr;
    if (learn_on) std::printf("[learn] observing; %zu queued from previous sessions\n",
                              lq.items.size());

    auto handle = [&](const std::string& line) -> std::string {
      // Gap B: repair a context-dependent turn by borrowing the previous turn's
      // tool and payload, then run the ordinary retrieval path on the repaired
      // sentence. The retrieval stack itself is untouched.
      const std::string query = sess.rewrite(line);
      if (dbg && query != line)
        std::printf("  [dbg] context: '%s' -> '%s'\n", line.c_str(), query.c_str());

      // -------------------------------------------------------------------
      //  Refuse a request that asks for two actions.
      //
      //  Measured: with no multi-action training data the system answers such a
      //  request by MIXING the two clauses -- "read the file X and list Y" becomes
      //  list_dir {"path":"X"}, i.e. the tool from the second clause with the
      //  payload from the first. That call is valid JSON, executes cleanly, and is
      //  wrong. Refusing with a reason is the recoverable outcome.
      //
      //  Checked on the RAW line, before context repair, because the repair step
      //  rewrites exactly the surface form this test looks for.
      // -------------------------------------------------------------------
      {
        size_t split = 0;
        if (detect_multi_action(line, &split)) {
          const std::string first  = line.substr(0, split);
          const std::string second = line.substr(split);
          std::string a = first, b = second;
          while (!a.empty() && a.back() == ' ') a.pop_back();
          size_t s = b.find_first_not_of(' ');
          if (s != std::string::npos) b = b.substr(s);
          return "this request asks for two actions, which I do not do in one turn.\n"
                 "  I read it as:  (1) " + a + "\n"
                 "                 (2) " + b + "\n"
                 "  Please send them one at a time.";
        }
      }

      std::vector<float> v; encode_hybrid(query, v);
      std::vector<std::pair<float,int>> top;
      int bi = query_reranked(mem, v, query, top);
      // The shared layer can refuse before retrieval runs; those returns are
      // negative but are NOT "empty memory", and reporting them as such would
      // replace a useful reason with a useless one.
      if (const char* rf = refusal_text(bi)) return rf;
      if (bi < 0) return "memory is empty";

      if (dbg) {
        std::printf("  [dbg] query        : '%s'\n", query.c_str());
        std::printf("  [dbg] query payload: '%s'\n", extract_payload(query).c_str());
        for (auto& t : top)
          std::printf("  [dbg]   cand %.3f : '%s'\n", t.first, mem.prompts[t.second].c_str());
        std::printf("  [dbg] matched      : '%s'\n", mem.prompts[bi].c_str());
      }
      // -------------------------------------------------------------------
      //  REFUSE WHEN NOTHING MATCHES WELL ENOUGH.
      //
      //  The memory always returns its nearest neighbour, and until now the
      //  system always acted on it. For requests outside its five tools that
      //  produced confident nonsense with real side effects:
      //
      //      "translate hello into french" -> write_file {"path":"french"}   CREATED A FILE
      //      "what is the weather in boston" -> list_dir {"path":"boston"}
      //      "flip a coin" -> calc {"expr":"on"}
      //      "set a timer for 5 minutes" -> list_dir {"path":"minutes"}
      //
      //  Seven of eight out-of-scope requests were answered with a fabricated
      //  tool call. Only one was recognised as conversation.
      //
      //  The similarity distributions separate cleanly, measured on real text:
      //      requests the tools CAN serve : 0.532 .. 1.000
      //      requests they cannot         : 0.110 .. 0.430
      //  so a floor in the gap is a principled boundary rather than a tuned knob.
      //
      //  A refusal is worth far more than a plausible wrong action: the user
      //  learns the limit immediately, and nothing is written to disk. This is the
      //  same principle as the bare-path default and the multi-action refusal.
      // -------------------------------------------------------------------
      // -------------------------------------------------------------------
      //  THE FLOOR MUST NOT DEPEND ON HOW LONG THE PAYLOAD IS.
      //
      //  Measured problem with a flat 0.48: a request the structure layer has
      //  already resolved was still refused when its payload happened to be long
      //  and distinctive.
      //
      //      "show me C:/Users/cc/canticle/STATUS.md"   -> 0.415  REFUSED
      //      "open C:/Users/cc/canticle/STATUS.md"      -> 0.343  REFUSED
      //      "what is the sum of 3 plus 5"              -> 0.445  REFUSED
      //      "what is the square root of 144"           -> 0.417  REFUSED
      //
      //  Every one of those is answerable, and all four were refused. The reason is
      //  that similarity is computed over the WHOLE sentence, so a longer or rarer
      //  path pushes the score down even though the REQUEST is perfectly clear. The
      //  stored neighbours use short paths like "C:/log/log.txt", so the more
      //  specific the user is, the less the system recognises them -- exactly
      //  backwards.
      //
      //  The floor was calibrated on short payloads, where it correctly separated
      //  in-scope requests (0.532+) from out-of-scope ones (0.43-). It is the wrong
      //  instrument for a sentence whose structure is unambiguous.
      //
      //  So the refusal is now conditional on the structure layer being UNSURE:
      //
      //      infer_intent(text) == Unknown   no rule claims this request, so the
      //                                      embedding is the only evidence and a
      //                                      weak match means refusal
      //      infer_intent(text) != Unknown   a rule has already decided the family
      //                                      and the payload was extracted, so the
      //                                      action is determined; only the CHOICE
      //                                      of stored example needs the embedding,
      //                                      and any plausible one will do
      //
      //  This keeps the behaviour that mattered -- "translate hello into french"
      //  and "what is the weather in boston" infer Unknown and are still refused --
      //  while no longer punishing a user for naming a full path.
      // -------------------------------------------------------------------
      // The floors are defined once, near the top of this file, because
      // query_reranked enforces them now and every entry point goes through it.

      const Intent want = infer_intent(query);

      // -------------------------------------------------------------------
      //  An operation with no tool gets its own answer.
      //
      //  Not the generic refusal: the difference between "I do not understand you"
      //  and "I understand, and I do not do that" is a real difference to the user,
      //  and stating which one applies is what lets them decide whether to reword
      //  or to stop asking.
      //
      //  Reached before any retrieval, because retrieval cannot help: there is no
      //  stored pattern for an operation this system has never had a tool for, so
      //  the nearest neighbour is guaranteed to be something else.
      // -------------------------------------------------------------------
      // -----------------------------------------------------------------------
      //  DELEGATED TO refusal_text, because this copy had drifted.
      //
      //  It said "Not available: delete, rename, copy, make a directory, file size,
      //  timestamps, searching file contents" -- while refusal_text(-4) said copy,
      //  mkdir, stat and search ARE available, because they are. Two entry points
      //  gave mutually exclusive answers about the same capability, and the suite
      //  built to catch exactly that (am-entries) only asserted that the reply
      //  CONTAINS "no tool for it", which both copies do.
      //
      //  A test that matches on a shared substring cannot distinguish two
      //  contradictory messages. The fix for that is not a better substring -- it is
      //  one message.
      // -----------------------------------------------------------------------
      if (want == Intent::Unsupported) return refusal_text(-4);

      // The confidence floor now lives in query_reranked and arrives as -5, so
      // every entry point applies the same one. See the note there.
      if (bi == -5) {
        const std::string matched = mem.prompts.empty() ? "" : mem.prompts[top[0].second];
        const Intent w2 = infer_intent(query);
        const float f2 = (w2 != Intent::Unknown) ? MIN_MATCH_STRUCTURED : MIN_MATCH_UNKNOWN;
        return "I do not have a tool for that.\n"
               "  The closest thing I know is \"" + matched + "\"\n"
               "  but it is not close enough to act on ("
               + std::to_string((int)(top[0].first * 100)) + "% match, below "
               + std::to_string((int)(f2 * 100)) + "%).\n"
               "  I can: read a file, list a directory, write a file, run a shell\n"
               "  command, or do arithmetic.";
      }

      std::string reply = slot_transfer(query, mem.prompts[bi], mem.replies[bi]);
      if (dbg) std::printf("  [dbg] after slot   : '%s'\n", reply.c_str());

      auto calls = parse_tool_calls(reply);
      const std::string tool = calls.empty() ? "" : calls[0].name;
      sess.push(line, tool, extract_payload(query));
      if (calls.empty()) return reply;           // plain chat turn

      // Structured dispatch so the outcome is known rather than inferred.
      const ToolResult tr = dispatch_tool_result(calls[0]);
      const std::string obs = tr.output;

      // ---- autonomous learning from this turn ----
      //
      // Two distinct signals, and they need different treatment:
      //
      //   SUCCESS  the action worked. It is worth reinforcing ONLY if the memory
      //            did not already answer the request well; otherwise the queue
      //            fills with paraphrases of what it already knows. The
      //            admission rule later checks that (SIM_CEILING).
      //
      //   FAILURE  the action did not work. Never reinforce it -- but remember
      //            it, because the user's next turn is usually the correction.
      //
      // RECORD THE USER'S OWN WORDS, NOT THE CONTEXT-REPAIRED QUERY.
      //
      // "query" above may have been rewritten by Session::rewrite to inherit the
      // previous turn's tool. That rewrite is correct for RETRIEVAL and wrong for
      // LEARNING: it turned the user's "play some blues" into
      // "run the command blues", so the queue held the system's internal phrasing
      // rather than the human phrasing, and frame induction over those records
      // found no shared prefix ("play some jazz" was followed by two
      // "run the command ..." entries).
      //
      // What must generalise is how USERS ask, because that is the text a future
      // retrieval actually receives.
      const std::string& learn_text = line;
      if (queue && getenv("AM_LEARN_DIAG")) {
        std::fprintf(stderr, "[ldbg] turn line='%s' query='%s' ok=%d have_failure_before=%d last_failed='%s'\n",
                     line.c_str(), query.c_str(), (int)tr.ok, (int)queue->have_failure,
                     queue->last_failed_raw.c_str());
      }
      if (queue) {
        if (tr.ok && !calls.empty()) {
          const bool paired = queue->pair_correction(learn_text, reply);
          if (getenv("AM_LEARN_DIAG"))
            std::fprintf(stderr, "[ldbg]   success: paired=%d\n", (int)paired);
          if (!paired) {
            const float sim = top.empty() ? 0.f : top[0].first;
            queue->observe(learn_text, reply, sim);
          } else if (dbg) {
            std::printf("  [dbg] paired correction: \"%s\" now answered by the action that worked\n",
                        queue->last_failed_raw.c_str());
          }
        } else if (!tr.ok) {
          queue->record_failure(learn_text, tr.error);
        }
      }
      return reply + "\n  -> " + obs + (tr.ok ? "" : "  [FAILED:" + tr.error + "]");
    };

    // Scripted multi-turn run: --script "turn1|turn2|turn3". Turns go through
    // the same Session as the REPL, so this exercises real inheritance.
    std::string script;
    for (int i = 1; i < argc; ++i)
      if (std::string(argv[i]) == "--script" && i + 1 < argc) script = argv[i + 1];

    // -----------------------------------------------------------------------
    //  Every exit path must flush the learning queue.
    //
    //  This was a real bug: the save lived only at the end of the interactive
    //  REPL, so --script and --prompt returned before it ran. A scripted session
    //  observed four turns, printed nothing about them, and left no queue file --
    //  the observations were discarded and the capability looked like it did not
    //  work at all. Exit paths are exactly where bookkeeping gets lost.
    // -----------------------------------------------------------------------
    auto finish = [&]() {
      if (!learn_on) return;
      lq.evaluate();
      lq.save();
      const auto acc = lq.accepted();
      std::printf("\n[learn] %zu observation(s), %zu admitted",
                  lq.items.size(), acc.size());
      if (lq.corrections) std::printf(", %d correction pairing(s)", lq.corrections);
      std::printf("\n");
      for (const auto& o : acc)
        std::printf("[learn]   ADMIT  hits=%d sim=%.2f  \"%s\"  (%s)\n",
                    o.hits, o.top_sim, o.query.c_str(), o.reason);
      if (!acc.empty())
        std::printf("[learn]   run: canticle am-harvest --apply\n");
    };

    if (!script.empty()) {
      std::vector<std::string> parts;
      std::string cur;
      for (char c : script) {
        if (c == '|') { parts.push_back(cur); cur.clear(); }
        else cur.push_back(c);
      }
      if (!cur.empty()) parts.push_back(cur);
      for (size_t i = 0; i < parts.size(); ++i) {
        double t0 = now_ms();
        std::string out = handle(parts[i]);
        double t1 = now_ms();
        std::printf("\n[%zu] user> %s\n    canticle> %s\n    [%.2f ms]\n",
                    i + 1, parts[i].c_str(), out.c_str(), t1 - t0);
      }
      finish();
      return 0;
    }

    if (!prompt.empty()) {
      double t0 = now_ms();
      std::string out = handle(prompt);
      double t1 = now_ms();
      std::printf("\nuser> %s\ncanticle> %s\n[%.2f ms]\n", prompt.c_str(), out.c_str(), t1 - t0);
      finish();
      return 0;
    }

    std::printf("type a message, Ctrl-Z then Enter to quit\n\n");
    std::string line;
    while (true) {
      std::printf("user> "); std::fflush(stdout);
      if (!std::getline(std::cin, line)) break;
      if (line.empty()) continue;
      double t0 = now_ms();
      std::string out = handle(line);
      double t1 = now_ms();
      std::printf("canticle> %s\n", out.c_str());
      std::printf("          [%.2f ms]\n\n", t1 - t0);
    }
    finish();
    return 0;
  }

  std::fprintf(stderr, "usage: canticle am-build|am-eval|am-ask|am-chat|am-bench [opts]\n");
  return 1;
}

namespace canticle {
// ---------------------------------------------------------------------------
//  Public query entry point for external harnesses (paraphrase tests, future
//  evaluation code). Loads the memory once and caches it, so repeated calls in
//  a loop do not pay the 106 MB read every time.
// ---------------------------------------------------------------------------
int am_query_public(const std::string& ampath, const std::string& prompt,
                    std::string& out_reply, float& out_sim) {
  static AmMem* mem = nullptr;
  static AmEncoder* enc = nullptr;
  static LexModel* lex = nullptr;
  static std::string loaded_path;

  if (!mem || loaded_path != ampath) {
    delete mem; delete enc; delete lex;
    mem = new AmMem();
    // The encoder must exist BEFORE loading, because the delta replay needs it
    // to encode the learned patterns. Creating it afterwards -- which is what an
    // earlier ordering did -- leaves the replay nothing to work with.
    if (!mem->load(ampath)) { delete mem; mem = nullptr; return 1; }
    enc = new AmEncoder();
    enc->D = mem->D;
    // IDF statistics were built at write time; rebuilding them here over the
    // stored prompts reproduces the same weighting closely enough for
    // retrieval, and keeps the file format free of encoder state.
    enc->build_idf(mem->prompts);
    lex = new LexModel();
    lex->DIM = HYB_LEX_DIM;
    {
      auto tr = synth_tool_data(4000, 99);
      std::vector<std::string> corpus;
      for (auto& s : tr) corpus.push_back(s.prompt);
      lex->train(corpus, HYB_LEX_DIM);
    }
    mem->replay_delta(ampath, *enc, *lex);
    loaded_path = ampath;
  }

  std::vector<float> v, lv, hk;
  enc->encode(prompt, v);
  lex->encode(prompt, lv);
  hybrid_key(v, lv, hk);
  (void)0;
  std::vector<std::pair<float,int>> top;
  const int bi = query_reranked(*mem, hk, prompt, top);
  if (const char* rf = refusal_text(bi)) {
    out_reply = rf;
    // The similarity is reported even on refusal: a caller measuring the refusal
    // boundary needs to know HOW far away the nearest neighbour was, and returning
    // zero would erase the very quantity that justified declining.
    out_sim = (bi == -5 && !top.empty()) ? top[0].first : 0.f;
    return 0;
  }
  if (bi < 0) return 1;
  out_reply = slot_transfer(prompt, mem->prompts[bi], mem->replies[bi]);
  out_sim = top.empty() ? 0.f : top[0].first;
  return 0;
}
} // namespace canticle
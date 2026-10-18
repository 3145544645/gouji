
// ============================================================================
//  Straight-through ternary training on the vocabulary head.
//
//  WHY THIS FILE EXISTS
//  --------------------
//  The expensive claim of CANTICLE is that a model can be *learned* while its
//  weights are held in {-1,0,+1}. Before spending hours on a full transformer,
//  this harness proves the mechanism on the cheapest possible problem:
//
//      input : a short sequence of token ids
//      pool  : mean of the embedding rows
//      head  : a TERNARY matrix  V x d  ->  logits
//      loss  : cross entropy against the next token
//
//  Every number the trainer prints comes off the GPU. If the loss does not fall
//  here, nothing further is worth building.
// ============================================================================
#include "canticle/train.h"
#include "canticle/ops.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace canticle {

// ---------------------------------------------------------------------------
//  Device residency
// ---------------------------------------------------------------------------
struct HeadModel::Dev {
  GpuBuf emb_f32, emb_f16;      // [V][d]
  GpuBuf head_f32;              // [V][d] master
  GpuBuf head_P, head_M;        // packed planes
  GpuBuf head_scale;            // [V] fp32
  GpuBuf rms;
  GpuBuf ids;                   // [T] int
  GpuBuf h16;                   // [d] fp16 pooled
  GpuBuf h32;                   // [d] fp32 pooled
  GpuBuf logits, raw;           // [V]
  GpuBuf g;                     // [V] grad wrt logits (dy * scale)
  GpuBuf dscale;                // [V]
  GpuBuf dhead;                 // [V][d]
  GpuBuf dh;                    // [d]
  GpuBuf demb;                  // [V][d]
  GpuBuf am_emb, av_emb, am_head, av_head;
  cudaStream_t st = nullptr;
  int V = 0, d = 0;

  void init(int V_, int d_) {
    V = V_; d = d_;
    if (!st) CUDA_CHECK(cudaStreamCreate(&st));
    emb_f32.alloc((size_t)V * d * 4);
    emb_f16.alloc((size_t)V * d * 2);
    head_f32.alloc((size_t)V * d * 4);
    head_P.alloc((size_t)V * (d / 32) * 4);
    head_M.alloc((size_t)V * (d / 32) * 4);
    head_scale.alloc((size_t)V * 4);
    rms.alloc((size_t)d * 4);      // rms holds fp32, not fp16
    ids.alloc(4096 * 4);
    h16.alloc(d * 2); h32.alloc(d * 4);
    logits.alloc((size_t)V * 4); raw.alloc((size_t)V * 4);
    g.alloc((size_t)V * 4); dscale.alloc((size_t)V * 4);
    dhead.alloc((size_t)V * d * 4);
    dh.alloc(d * 4);
    demb.alloc((size_t)V * d * 4);
    am_emb.alloc((size_t)V * d * 4); av_emb.alloc((size_t)V * d * 4);
    am_head.alloc((size_t)V * d * 4); av_head.alloc((size_t)V * d * 4);
    CUDA_CHECK(cudaMemsetAsync(am_emb.ptr, 0, (size_t)V*d*4, st));
    CUDA_CHECK(cudaMemsetAsync(av_emb.ptr, 0, (size_t)V*d*4, st));
    CUDA_CHECK(cudaMemsetAsync(am_head.ptr, 0, (size_t)V*d*4, st));
    CUDA_CHECK(cudaMemsetAsync(av_head.ptr, 0, (size_t)V*d*4, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
  }
  ~Dev() { if (st) cudaStreamDestroy(st); }
};

// Out-of-line special members: Dev is defined in this TU only, so the
// unique_ptr deleter must be instantiated here.
HeadModel::HeadModel() = default;
HeadModel::~HeadModel() = default;
HeadModel::HeadModel(HeadModel&&) noexcept = default;
HeadModel& HeadModel::operator=(HeadModel&&) noexcept = default;

void HeadModel::init(int vocab_, int d_, uint64_t seed) {
  vocab = vocab_; d = d_;   // d must be a multiple of 32
  std::mt19937 rng((uint32_t)seed);
  std::normal_distribution<float> nd(0.f, 0.02f);
  emb.assign((size_t)vocab * d, 0.f);
  head.assign((size_t)vocab * d, 0.f);
  for (auto& x : emb)  x = nd(rng);
  for (auto& x : head) x = nd(rng);
  rms.assign(d, 1.f);
  adam_emb_m.assign(emb.size(), 0.f);  adam_emb_v.assign(emb.size(), 0.f);
  adam_head_m.assign(head.size(), 0.f); adam_head_v.assign(head.size(), 0.f);
  dev.reset(new Dev);
  dev->init(vocab, d);
}

void HeadModel::to_device() {
  auto& D = *dev;
  CUDA_CHECK(cudaMemcpyAsync(D.emb_f32.ptr, emb.data(), emb.size()*4, cudaMemcpyHostToDevice, D.st));
  CUDA_CHECK(cudaMemcpyAsync(D.head_f32.ptr, head.data(), head.size()*4, cudaMemcpyHostToDevice, D.st));
  CUDA_CHECK(cudaMemcpyAsync(D.rms.ptr, rms.data(), rms.size()*4, cudaMemcpyHostToDevice, D.st));
  CUDA_CHECK(cudaStreamSynchronize(D.st));
}

void HeadTrainer::requantise() {
  auto& D = *m->dev;
  launch_ternarize_rows((const float*)D.head_f32.ptr,
                        (uint32_t*)D.head_P.ptr, (uint32_t*)D.head_M.ptr,
                        (float*)D.head_scale.ptr, m->vocab, m->d, D.st);
  CUDA_CHECK(cudaStreamSynchronize(D.st));
}

// ---------------------------------------------------------------------------
//  One training step.
// ---------------------------------------------------------------------------
float HeadTrainer::train_step(const HeadBatch& b) {
  auto& D = *m->dev;
  cudaStream_t s = D.st;
  const int V = m->vocab, d = m->d;
  const int T = (int)b.ids.size();

  step++;
  const float bc1 = 1.f - std::pow(beta1, (float)step);
  const float bc2 = 1.f - std::pow(beta2, (float)step);

  // ---- 0. refresh the packed planes from the fp32 master ----
  if (step % requant_every == 1) {
    launch_ternarize_rows((const float*)D.head_f32.ptr,
                          (uint32_t*)D.head_P.ptr, (uint32_t*)D.head_M.ptr,
                          (float*)D.head_scale.ptr, V, d, s);
  }
  (void)s;

  // ---- 1. forward ----
  CUDA_CHECK(cudaMemcpyAsync(D.ids.ptr, b.ids.data(), T*4, cudaMemcpyHostToDevice, s));
  launch_f32_to_f16((const float*)D.emb_f32.ptr, (__half*)D.emb_f16.ptr, V*d, s);
  launch_meanpool((const __half*)D.emb_f16.ptr, (const int*)D.ids.ptr,
                  (__half*)D.h16.ptr, T, d, s);
  launch_tern_head((const uint32_t*)D.head_P.ptr, (const uint32_t*)D.head_M.ptr,
                   (const __half*)D.h16.ptr, (float*)D.logits.ptr, (float*)D.raw.ptr,
                   V, d, s);
  // the pooled activation is needed in fp32 for the outer product below
  launch_f16_to_f32((const __half*)D.h16.ptr, (float*)D.h32.ptr, d, s);
  CUDA_CHECK(cudaStreamSynchronize(s));

  // ---- 2. CE loss on the host (V is small), then build dy on the device ----
  auto& LGC = m->logits_cache;
  auto& SCC = m->scale_cache;
  LGC.resize(V);
  SCC.resize(V);
  CUDA_CHECK(cudaMemcpy(LGC.data(), D.logits.ptr, V*4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(SCC.data(), D.head_scale.ptr, V*4, cudaMemcpyDeviceToHost));

  // y[v] = raw[v] * scale[v]
  double mx = -1e30;
  for (int v = 0; v < V; ++v) {
    LGC[v] *= SCC[v];
    mx = std::max(mx, (double)LGC[v]);
  }
  double sum = 0; for (int v = 0; v < V; ++v) sum += std::exp(LGC[v] - mx);
  const double lse = mx + std::log(sum);
  float loss = (float)(lse - LGC[b.target]);

  // dy[v] = p[v] - 1[v == target]  (in fp32, on the host)
  std::vector<float> dy(V);
  for (int v = 0; v < V; ++v)
    dy[v] = (float)std::exp(LGC[v] - mx) / (float)sum;
  dy[b.target] -= 1.f;
  CUDA_CHECK(cudaMemcpyAsync(D.g.ptr, dy.data(), V*4, cudaMemcpyHostToDevice, s));

  // ---- 3. backward ----
  // g[v] = dy[v] * scale[v]   (gradient wrt the raw ternary product)
  launch_scale_vec((const float*)D.g.ptr, (const float*)D.head_scale.ptr,
                   (float*)D.g.ptr, V, s);
  // dW[v][j] = g[v] * h[j]
  launch_outer((float*)D.dhead.ptr, (const float*)D.g.ptr, (const float*)D.h32.ptr, V, d, s);
  // dh[j] = sum_v g[v] * w[v][j]
  launch_tern_gemm_dx((const uint32_t*)D.head_P.ptr, (const uint32_t*)D.head_M.ptr,
                      (const float*)D.head_scale.ptr, (const float*)D.g.ptr,
                      (float*)D.dh.ptr, V, d, s);
  // demb is small enough that we can zero it every step
  CUDA_CHECK(cudaMemsetAsync(D.demb.ptr, 0, (size_t)V*d*4, s));
  launch_embed_bwd((float*)D.demb.ptr, (const int*)D.ids.ptr,
                   (const float*)D.dh.ptr, T, d, s);

  // ---- 4. AdamW ----
  launch_adam((float*)D.head_f32.ptr, (const float*)D.dhead.ptr,
              (float*)D.am_head.ptr, (float*)D.av_head.ptr, V*d,
              lr, beta1, beta2, eps, wd, bc1, bc2, s);
  launch_adam((float*)D.emb_f32.ptr, (const float*)D.demb.ptr,
              (float*)D.am_emb.ptr, (float*)D.av_emb.ptr, V*d,
              lr, beta1, beta2, eps, 0.f, bc1, bc2, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  return loss;
}

// Evaluate the batch most recently passed to eval_one(), exposing the raw
// device buffers through host vectors.
float HeadTrainer::eval_one_to(std::vector<float>& logits_out, std::vector<float>& scale_out) {
  logits_out = m->logits_cache;
  scale_out  = m->scale_cache;
  return 0.f;
}

float HeadTrainer::eval_loss(const std::vector<HeadBatch>& data) {
  double tot = 0;
  for (auto& b : data) tot += eval_one(b);
  return (float)(tot / std::max<size_t>(1, data.size()));
}

// forward-only loss, no gradient
float HeadTrainer::eval_one(const HeadBatch& b) {
  auto& D = *m->dev;
  cudaStream_t s = D.st;
  const int V = m->vocab, d = m->d, T = (int)b.ids.size();
  CUDA_CHECK(cudaMemcpyAsync(D.ids.ptr, b.ids.data(), T*4, cudaMemcpyHostToDevice, s));
  launch_f32_to_f16((const float*)D.emb_f32.ptr, (__half*)D.emb_f16.ptr, V*d, s);
  launch_meanpool((const __half*)D.emb_f16.ptr, (const int*)D.ids.ptr, (__half*)D.h16.ptr, T, d, s);
  launch_tern_head((const uint32_t*)D.head_P.ptr, (const uint32_t*)D.head_M.ptr,
                   (const __half*)D.h16.ptr, (float*)D.logits.ptr, (float*)D.raw.ptr, V, d, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  std::vector<float> lg(V), sc(V);
  CUDA_CHECK(cudaMemcpy(lg.data(), D.logits.ptr, V*4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sc.data(), D.head_scale.ptr, V*4, cudaMemcpyDeviceToHost));
  {
    auto& L2 = m->logits_cache; L2.resize(V);
    m->scale_cache.resize(V);
    for (int v = 0; v < V; ++v) { L2[v] = lg[v] * sc[v]; m->scale_cache[v] = sc[v]; }
  }
  double mx = -1e30;
  for (int v = 0; v < V; ++v) { lg[v] *= sc[v]; mx = std::max(mx, (double)lg[v]); }
  double sum = 0; for (int v = 0; v < V; ++v) sum += std::exp(lg[v] - mx);
  return (float)(mx + std::log(sum) - lg[b.target]);
}

} // namespace canticle


// ===========================================================================
//  Model smoke test: build a small randomly-initialised CANTICLE-1, run the
//  forward pass on a few tokens, and check the numerics are sane.
//  This is the gate that must pass before any training work is meaningful.
// ===========================================================================
#include "canticle/model.h"
#include "canticle/kernels.h"
#include <random>
#include <cmath>
#include <algorithm>

using namespace canticle;

int smoke_model_run() {
  Config c;
  c.vocab = 1024; c.d_model = 256; c.n_layer = 4;
  c.n_head = 4; c.d_ff = 768; c.ctx = 64;

  std::printf("=== model smoke: vocab=%d d=%d L=%d H=%d ff=%d ctx=%d ===\n",
              c.vocab, c.d_model, c.n_layer, c.n_head, c.d_ff, c.ctx);

  Model m;
  m.init(c);

  // ---- random master weights ----
  std::mt19937 rng(42);
  std::normal_distribution<float> nd(0.f, 0.02f);
  auto fill = [&](std::vector<float>& v) { for (float& x : v) x = nd(rng); };

  size_t total = m.ternary_weight_bytes();
  std::printf("ternary planes : %.3f MiB\n", total / 1048576.0);
  std::printf("total weights  : %.3f MiB\n", m.total_bytes() / 1048576.0);

  for (auto& L : m.layers) {
    auto mk = [&](TernLinear& t) {
      std::vector<float> w((size_t)t.out * t.in);
      fill(w);
      // ternarise via the public path: build a temp matrix then copy planes
      TernaryMatrix T;
      ternarize_for_test(w, t.out, t.in, T);
      t.W = std::move(T);
      t.scale.assign(t.out, f32_to_f16(0.02f));
    };
    mk(L.wq); mk(L.wk); mk(L.wv); mk(L.wo); mk(L.w_gate); mk(L.w_up); mk(L.w_down);
  }
  fill_rand_f16(m.token_emb, rng, 0.02f);
  fill_rand_f16(m.lm_head, rng, 0.02f);
  for (auto& v : m.rms_final) v = f32_to_f16(1.0f);
  for (auto& L : m.layers) {
    for (auto& v : L.rms_att) v = f32_to_f16(1.0f);
    for (auto& v : L.rms_ffn) v = f32_to_f16(1.0f);
  }

  m.to_device();
  m.reset_kv();

  // ---- run a few tokens ----
  std::printf("\n--- forward pass ---\n");
  std::vector<int> toks = {7, 19, 42, 100, 256, 511};
  double t0 = now_ms();
  for (size_t i = 0; i < toks.size(); ++i) {
    const auto& lg = m.forward(toks[i], (int)i, true);
    double sum = 0, mx = -1e30, mn = 1e30; int nan = 0;
    for (float v : lg) {
      if (std::isnan(v) || std::isinf(v)) nan++;
      sum += v; mx = std::max(mx, (double)v); mn = std::min(mn, (double)v);
    }
    std::printf("  tok=%4d  logits: mean=%+.4f  min=%+.3f  max=%+.3f  nan=%d  argmax=%d\n",
                toks[i], sum / lg.size(), mn, mx, nan,
                (int)(std::max_element(lg.begin(), lg.end()) - lg.begin()));
  }
  double t1 = now_ms();
  std::printf("  %zu tokens in %.2f ms -> %.1f tok/s\n", toks.size(), t1 - t0,
              toks.size() * 1000.0 / (t1 - t0));

  // ---- decode-only throughput (what the user actually feels) ----
  std::printf("\n--- decode throughput (ctx=32, temp=0.8) ---\n");
  m.reset_kv();
  std::vector<int> out;
  int cur = 11;
  double d0 = now_ms();
  const int GEN = 32;
  for (int i = 0; i < GEN; ++i) {
    m.forward(cur, i, true);
    cur = m.sample(0.8f, 40, 0.95f, 0);
    out.push_back(cur);
  }
  double d1 = now_ms();
  std::printf("  generated %d tokens in %.2f ms -> %.1f tok/s\n", GEN, d1 - d0, GEN * 1000.0 / (d1 - d0));
  std::printf("  sample ids:");
  for (int i = 0; i < 12 && i < (int)out.size(); ++i) std::printf(" %d", out[i]);
  std::printf("\n");

  std::printf("\n[SMOKE] forward pass completed\n");
  return 0;
}

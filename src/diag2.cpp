
#include "canticle/ftrain.h"
#include "canticle/ckpt.h"
#include "canticle/data.h"
#include "canticle/ops.h"
#include <cstdio>
#include <cmath>
#include <random>
using namespace canticle;

// Reuse the trainer plumbing but stop at the first non-finite loss.
int diag2_run() {
  std::printf("=== NaN hunt ===\n");
  auto samples = synth_tool_data(200, 7);
  std::string corpus;
  for (auto& s : samples) corpus += "<|user|>\n" + s.prompt + "\n<|assistant|>\n" + s.reply + "\n";
  BPETokenizer tk; tk.train(corpus, 1024);
  Model m;
  Config c; c.vocab = (int)tk.vocab.size(); c.d_model = 128; c.n_layer = 2;
  c.n_head = 2; c.d_ff = 384; c.ctx = 128;
  m.init(c);
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0.f, 0.04f);
  for (auto& L : m.layers) {
    auto mk = [&](TernLinear& t) {
      for (int n = 0; n < t.out; ++n) {
        std::vector<float> w(t.in); double as = 0;
        for (int k = 0; k < t.in; ++k) { w[k] = nd(rng); as += std::fabs(w[k]); }
        float thr = (float)(0.7*as/t.in); double keep = 0; int nk = 0;
        for (int k = 0; k < t.in; ++k) {
          uint32_t bit = 1u << (k&31); size_t wi = (size_t)n*(t.in/32) + (k>>5);
          if (w[k] >  thr) { t.W.P[wi] |= bit; keep += std::fabs(w[k]); nk++; }
          if (w[k] < -thr) { t.W.M[wi] |= bit; keep += std::fabs(w[k]); nk++; }
        }
        t.scale[n] = f32_to_f16(nk ? (float)(keep/nk) : 1e-3f);
      }
    };
    mk(L.wq); mk(L.wk); mk(L.wv); mk(L.wo); mk(L.w_gate); mk(L.w_up); mk(L.w_down);
  }
  for (auto& v : m.token_emb) v = f32_to_f16(nd(rng));
  for (auto& v : m.lm_head)   v = f32_to_f16(nd(rng));
  for (auto& L : m.layers) { for (auto& v : L.rms_att) v = f32_to_f16(1.f);
                             for (auto& v : L.rms_ffn) v = f32_to_f16(1.f); }
  for (auto& v : m.rms_final) v = f32_to_f16(1.f);

  // ---- inspect the weights the trainer will read ----
  auto stat = [&](const char* tag, const std::vector<uint16_t>& v) {
    int nan = 0; double mx = 0;
    for (uint16_t h : v) { float f = f16_to_f32(h); if (f != f) nan++; mx = std::max(mx, (double)std::fabs(f)); }
    std::printf("  %-12s n=%zu nan=%d maxabs=%.4f\n", tag, v.size(), nan, mx);
  };
  stat("token_emb", m.token_emb);
  stat("lm_head", m.lm_head);
  stat("rms_final", m.rms_final);
  stat("wq.scale", m.layers[0].wq.scale);
  stat("wq.P", std::vector<uint16_t>((uint16_t*)m.layers[0].wq.W.P.data(),
       (uint16_t*)m.layers[0].wq.W.P.data() + m.layers[0].wq.W.P.size()*2));
  // weight byte sanity
  for (int j = 0; j < 7; ++j) {
    TernLinear* t = (TernLinear*)(&m.layers[0].wq) + j;
  }
  std::printf("  wq: out=%d in=%d P.words=%zu expect=%d\n",
              m.layers[0].wq.out, m.layers[0].wq.in,
              m.layers[0].wq.W.P.size(), m.layers[0].wq.out * (m.layers[0].wq.in/32));
  std::printf("  wd: out=%d in=%d P.words=%zu expect=%d\n",
              m.layers[0].w_down.out, m.layers[0].w_down.in,
              m.layers[0].w_down.W.P.size(), m.layers[0].w_down.out * (m.layers[0].w_down.in/32));
  std::printf("  ff=%d  d=%d\n", m.cfg.d_ff, m.cfg.d_model);

  // ---- check the ternary plane bit content actually decodes ----
  {
    int pos = 0, neg = 0, zero = 0;
    auto& T = m.layers[0].wq;
    for (int n = 0; n < T.out && n < 4; ++n)
      for (int k = 0; k < T.in; ++k) {
        size_t wi = (size_t)n*(T.in/32) + (k>>5); uint32_t bit = 1u << (k&31);
        int w = (int)((T.W.P[wi]&bit)!=0) - (int)((T.W.M[wi]&bit)!=0);
        if (w > 0) pos++; else if (w < 0) neg++; else zero++;
      }
    std::printf("  wq row0..3 ternary: +1=%d  -1=%d  0=%d\n", pos, neg, zero);
  }
  std::printf("[diag] done\n");
  return 0;
}

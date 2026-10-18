
// ============================================================================
//  Checkpoint I/O. One flat little-endian binary file, no dependencies.
//  Layout:  magic "CANT" | version | config(7 x int32) | tokenizer blob
//           | token_emb fp16 | lm_head fp16 | rms_final fp16
//           | per layer: 7 x (P,M planes + fp16 scales) + 2 rms tables
// ============================================================================
#include "canticle/ckpt.h"
#include <cstdio>
#include <cstring>

namespace canticle {

static void w32(FILE* f, int32_t v) { std::fwrite(&v, 4, 1, f); }
static void w64(FILE* f, uint64_t v) { std::fwrite(&v, 8, 1, f); }
static int32_t r32(FILE* f) { int32_t v = 0; std::fread(&v, 4, 1, f); return v; }
static uint64_t r64(FILE* f) { uint64_t v = 0; std::fread(&v, 8, 1, f); return v; }

template <typename T>
static void wvec(FILE* f, const std::vector<T>& v) {
  uint64_t n = v.size(); w64(f, n);
  if (n) std::fwrite(v.data(), sizeof(T), n, f);
}
template <typename T>
static void rvec(FILE* f, std::vector<T>& v) {
  uint64_t n = r64(f); v.resize(n);
  if (n) std::fread(v.data(), sizeof(T), n, f);
}

bool save_checkpoint(const std::string& path, const Model& m, const BPETokenizer* tk) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  std::fwrite("CANT", 4, 1, f);
  w32(f, 1);
  const Config& c = m.cfg;
  w32(f, c.vocab); w32(f, c.d_model); w32(f, c.n_layer);
  w32(f, c.n_head); w32(f, c.d_ff);   w32(f, c.ctx);

  // ---- tokenizer ----
  // LENGTH-PREFIXED, not newline-delimited.
  //
  // A byte-level BPE vocabulary contains entries that ARE the newline byte
  // (vocab[6] is "\n" -- the 256 raw bytes are the base alphabet). A
  // line-based text format therefore cannot represent the vocabulary at all:
  // one newline entry shifts every following id, and after a save/load round
  // trip <|user|> moved from id 2 to id 792 with merges silently becoming 0.
  // Inference then tokenised a different way than training and every reply
  // collapsed to a single repeated token.
  uint64_t has_tk = tk ? 1 : 0;
  w64(f, has_tk);
  if (tk) {
    uint64_t nv = tk->vocab.size(); w64(f, nv);
    for (auto& s : tk->vocab) {
      uint32_t len = (uint32_t)s.size();
      std::fwrite(&len, 4, 1, f);
      if (len) std::fwrite(s.data(), 1, len, f);
    }
    uint64_t nm = tk->merges.size(); w64(f, nm);
    for (auto& pr : tk->merges) {
      int32_t a = pr.first, b = pr.second;
      std::fwrite(&a, 4, 1, f); std::fwrite(&b, 4, 1, f);
    }
  }

  wvec(f, m.token_emb);
  wvec(f, m.lm_head);
  wvec(f, m.rms_final);
  for (const auto& L : m.layers) {
    const TernLinear* ls[7] = {&L.wq,&L.wk,&L.wv,&L.wo,&L.w_gate,&L.w_up,&L.w_down};
    for (int j = 0; j < 7; ++j) {
      w32(f, ls[j]->out); w32(f, ls[j]->in);
      wvec(f, ls[j]->W.P); wvec(f, ls[j]->W.M); wvec(f, ls[j]->scale);
    }
    wvec(f, L.rms_att); wvec(f, L.rms_ffn);
  }
  std::fclose(f);
  return true;
}

bool load_checkpoint(const std::string& path, Model& m, BPETokenizer* tk) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[4] = {0}; std::fread(magic, 4, 1, f);
  if (std::memcmp(magic, "CANT", 4) != 0) { std::fclose(f); return false; }
  int32_t ver = r32(f); (void)ver;
  Config c;
  c.vocab = r32(f); c.d_model = r32(f); c.n_layer = r32(f);
  c.n_head = r32(f); c.d_ff = r32(f); c.ctx = r32(f);
  m.init(c);

  uint64_t has_tk = r64(f);
  if (has_tk && tk) {
    uint64_t nv = r64(f);
    tk->vocab.resize(nv);
    for (uint64_t i = 0; i < nv; ++i) {
      uint32_t len = 0;
      std::fread(&len, 4, 1, f);
      tk->vocab[i].resize(len);
      if (len) std::fread(&tk->vocab[i][0], 1, len, f);
    }
    uint64_t nm = r64(f);
    tk->merges.resize(nm);
    for (uint64_t i = 0; i < nm; ++i) {
      int32_t a = 0, b = 0;
      std::fread(&a, 4, 1, f); std::fread(&b, 4, 1, f);
      tk->merges[i] = {a, b};
    }
    tk->enc.clear();
    for (size_t i = 0; i < tk->vocab.size(); ++i) tk->enc[tk->vocab[i]] = (int)i;
    tk->merge_rank.clear();
    for (size_t i = 0; i < tk->merges.size(); ++i)
      tk->merge_rank[BPETokenizer::key(tk->merges[i].first, tk->merges[i].second)] = (int)i;
  }

  rvec(f, m.token_emb);
  rvec(f, m.lm_head);
  rvec(f, m.rms_final);
  for (auto& L : m.layers) {
    TernLinear* ls[7] = {&L.wq,&L.wk,&L.wv,&L.wo,&L.w_gate,&L.w_up,&L.w_down};
    for (int j = 0; j < 7; ++j) {
      int o = r32(f), in = r32(f);
      ls[j]->alloc(o, in);
      rvec(f, ls[j]->W.P); rvec(f, ls[j]->W.M); rvec(f, ls[j]->scale);
    }
    rvec(f, L.rms_att); rvec(f, L.rms_ffn);
  }
  std::fclose(f);
  return true;
}

} // namespace canticle

// ============================================================================
//  Learned lexical embedding -- implementation.
//
//  Count-based distributional semantics: PPMI matrix, then a truncated SVD by
//  power iteration on M M^T. Everything here is plain C++ over the same corpus
//  the memory is built from; there is no training loop and nothing to converge.
//
//  Why SVD by power iteration rather than something cleverer: the vocabulary is
//  a few thousand words and the target rank is 160, so a handful of iterations
//  gets the top singular directions to more precision than a mean-vector
//  sentence representation can use. Measured runtime is milliseconds.
// ============================================================================
#include "canticle/lex.h"
#include <cstdio>
#include <random>

namespace canticle {

// ---------------------------------------------------------------------------
//  A word and its context are both drawn from the same vocabulary. Contexts are
//  the words within a symmetric window; a window of 4 is wide enough to connect
//  a verb to the noun it governs across short commands like
//  "please run the command hostname".
// ---------------------------------------------------------------------------
static const int WINDOW = 4;

void LexModel::train(const std::vector<std::string>& corpus, int dim) {
  DIM = dim;

  // ---- vocabulary and co-occurrence counts ----
  std::vector<std::vector<int>> toks;
  toks.reserve(corpus.size());
  for (const auto& line : corpus) {
    auto t = lex_tokens(line);
    std::vector<int> ids;
    for (const auto& w : t) {
      auto it = w2i.find(w);
      int i;
      if (it == w2i.end()) { i = (int)words.size(); w2i[w] = i; words.push_back(w); }
      else i = it->second;
      ids.push_back(i);
    }
    toks.push_back(std::move(ids));
  }
  const int V = (int)words.size();
  if (V == 0) return;

  // co-occurrence as a hash map, since the matrix is sparse in practice
  std::unordered_map<uint64_t, float> co;
  co.reserve((size_t)V * 64);
  std::vector<double> wcount(V, 0.0);
  double total = 0.0;
  for (const auto& ids : toks) {
    const int L = (int)ids.size();
    for (int i = 0; i < L; ++i) {
      wcount[ids[i]] += 1.0;
      total += 1.0;
      for (int d = -WINDOW; d <= WINDOW; ++d) {
        if (d == 0) continue;
        const int j = i + d;
        if (j < 0 || j >= L) continue;
        // distance weighting: nearer words matter more
        const float wgt = 1.0f / (float)std::abs(d);
        co[((uint64_t)ids[i] << 32) | (uint32_t)ids[j]] += wgt;
      }
    }
  }
  if (total <= 0) return;
  std::vector<double> ccount(V, 0.0);
  for (auto& kv : co) ccount[(uint32_t)(kv.first & 0xffffffffu)] += kv.second;

  // ---- build a dense PPMI matrix (V x V) ----
  // V is a few thousand, so V*V floats is a few tens of MB at worst.
  std::vector<float> M((size_t)V * V, 0.f);
  for (auto& kv : co) {
    const int w = (int)(kv.first >> 32);
    const int c = (int)(kv.first & 0xffffffffu);
    const double pmi = std::log((kv.second * total) / (wcount[w] * ccount[c] + 1e-9) + 1e-9);
    if (pmi > 0) M[(size_t)w * V + c] = (float)pmi;
  }

  // ---- truncated SVD by power iteration on M M^T ----
  // We only need U and S, because vec(w) = U(w,:) * sqrt(S).
  std::mt19937 rng(12345);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> Q((size_t)V * DIM);
  for (auto& q : Q) q = nd(rng);
  std::vector<float> Sv(DIM, 0.f);

  const int ITERS = 12;   // enough for the leading directions at this scale
  std::vector<float> tmp(V);
  for (int it = 0; it < ITERS; ++it) {
    for (int k = 0; k < DIM; ++k) {
      // tmp = M * M^T * q_k
      std::vector<float> mq(V, 0.f);
      for (int w = 0; w < V; ++w) {
        double s = 0;
        const float* row = &M[(size_t)w * V];
        for (int c = 0; c < V; ++c) if (row[c] != 0.f) s += (double)row[c] * Q[(size_t)c * DIM + k];
        mq[w] = (float)s;
      }
      for (int w = 0; w < V; ++w) {
        double s = 0;
        const float* row = &M[(size_t)w * V];
        for (int c = 0; c < V; ++c) if (row[c] != 0.f) s += (double)row[c] * mq[c];
        tmp[w] = (float)s;
      }
      // deflate against previously computed directions
      for (int j = 0; j < k; ++j) {
        double dot = 0;
        for (int w = 0; w < V; ++w) dot += (double)tmp[w] * Q[(size_t)w * DIM + j];
        for (int w = 0; w < V; ++w) tmp[w] -= (float)dot * Q[(size_t)w * DIM + j];
      }
      double nrm = 0;
      for (int w = 0; w < V; ++w) nrm += (double)tmp[w] * tmp[w];
      nrm = std::sqrt(nrm);
      if (nrm < 1e-12) continue;
      for (int w = 0; w < V; ++w) Q[(size_t)w * DIM + k] = tmp[w] / (float)nrm;
    }
  }

  // singular values: |M^T q_k| gives the Rayleigh quotient for M M^T
  for (int k = 0; k < DIM; ++k) {
    std::vector<float> mq(V, 0.f);
    for (int w = 0; w < V; ++w) {
      double s = 0;
      const float* row = &M[(size_t)w * V];
      for (int c = 0; c < V; ++c) if (row[c] != 0.f) s += (double)row[c] * Q[(size_t)c * DIM + k];
      mq[w] = (float)s;
    }
    double nrm = 0;
    for (int c = 0; c < V; ++c) nrm += (double)mq[c] * mq[c];
    Sv[k] = (float)std::sqrt(std::sqrt(nrm));   // 4th root: M M^T eigenvalue -> singular value
  }

  // vec(w) = U(w,k) * sqrt(S_k)
  emb.assign((size_t)V * DIM, 0.f);
  for (int w = 0; w < V; ++w)
    for (int k = 0; k < DIM; ++k)
      emb[(size_t)w * DIM + k] = Q[(size_t)w * DIM + k] * std::sqrt(std::max(0.f, Sv[k]));
}

// ---------------------------------------------------------------------------
//  Sentence vector = mean of its word vectors, L2 normalised.
//  Mean rather than sum so that sentence length does not dominate the geometry,
//  and skipping unknown words rather than hashing them so that a sentence full
//  of unseen vocabulary scores a low coverage instead of a random direction.
// ---------------------------------------------------------------------------
void LexModel::encode(const std::string& text, std::vector<float>& out) const {
  out.assign(DIM, 0.f);
  auto t = lex_tokens(text);
  int used = 0;
  for (const auto& w : t) {
    const int i = id(w);
    if (i < 0) continue;
    const float* v = &emb[(size_t)i * DIM];
    for (int k = 0; k < DIM; ++k) out[k] += v[k];
    used++;
  }
  if (used == 0) return;
  for (float& x : out) x /= (float)used;

  double nrm = 0;
  for (float x : out) nrm += (double)x * x;
  nrm = std::sqrt(nrm);
  if (nrm > 1e-9) for (float& x : out) x /= (float)nrm;
}

float LexModel::coverage(const std::string& text) const {
  auto t = lex_tokens(text);
  if (t.empty()) return 0.f;
  int known = 0;
  for (const auto& w : t) if (id(w) >= 0) known++;
  return (float)known / (float)t.size();
}

// ---------------------------------------------------------------------------
//  Persistence. Length-prefixed, binary, little-endian -- the same discipline
//  used for the tokenizer, and for the same reason: a text format cannot safely
//  carry arbitrary content (the tokenizer's vocabulary contains raw newlines).
// ---------------------------------------------------------------------------
bool LexModel::save(std::ostream& f) const {
  int32_t hdr[2] = {DIM, (int)words.size()};
  f.write((const char*)hdr, 8);
  for (const auto& w : words) {
    uint32_t n = (uint32_t)w.size();
    f.write((const char*)&n, 4);
    if (n) f.write(w.data(), n);
  }
  uint64_t ne = emb.size();
  f.write((const char*)&ne, 8);
  if (ne) f.write((const char*)emb.data(), (std::streamsize)(ne * sizeof(float)));
  return (bool)f;
}

bool LexModel::load(std::istream& f) {
  int32_t hdr[2] = {0, 0};
  f.read((char*)hdr, 8);
  if (!f) return false;
  DIM = hdr[0];
  const int n = hdr[1];
  words.resize(n);
  for (int i = 0; i < n; ++i) {
    uint32_t len = 0;
    f.read((char*)&len, 4);
    words[i].resize(len);
    if (len) f.read(&words[i][0], len);
  }
  w2i.clear();
  for (int i = 0; i < n; ++i) w2i[words[i]] = i;
  uint64_t ne = 0;
  f.read((char*)&ne, 8);
  emb.resize(ne);
  if (ne) f.read((char*)emb.data(), (std::streamsize)(ne * sizeof(float)));
  return (bool)f;
}

} // namespace canticle

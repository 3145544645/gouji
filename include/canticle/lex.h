#pragma once
// ===========================================================================
//  Learned lexical embedding.
//
//  (#pragma once added after a real build failure: without it, a translation unit
//   that includes this header twice -- directly and through another header --
//   redefined lex_tokens, HYB_GRAM_DIM and LexModel, and the build stopped with
//   C2084/C2011. Every header in this project needs a guard.)
//
//  MEASURED PROBLEM
//  ----------------
//  am-sim showed the character n-gram encoder has essentially NO semantic
//  structure: 84 of 90 cross-intent pairs were closer than the worst
//  same-intent pair, and same-intent cosine for calc was 0.0020. Character
//  trigrams measure surface overlap, and "crunch 31 + 47" / "tally 62 - 18"
//  share almost no characters, so they land orthogonal no matter how similar
//  their meaning.
//
//  WHAT THIS FILE DOES INSTEAD
//  ---------------------------
//  Learn a vector per WORD from how words co-occur across the corpus, then
//  represent a sentence as the mean of its word vectors. Two sentences are
//  similar when their WORDS are used alike -- which is exactly what makes
//  "fire off hostname" and "execute hostname" land near each other despite
//  sharing no characters in the verb.
//
//  The method is PPMI + truncated SVD, both implemented here, on vocabulary
//  counts from the same corpus the memory is built from. No training loop, no
//  gradients, no framework: this is count-based distributional semantics, the
//  classical method that predates embeddings-by-backprop and needs far less data.
//
//    PPMI(w,c) = max(0, log( (count(w,c) * total) / (count(w) * count(c)) ))
//    M         = PPMI matrix, |V| x |C|
//    M         ~= U S V^T  via power iteration on M M^T
//    vec(w)    = row of U scaled by sqrt(S)
//
//  Dimensionality is deliberately small (160) so the vectors can be compared
//  with a plain dot product against the 2048-wide sparse keys.
// ===========================================================================
#include "canticle/am.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <iostream>

namespace canticle {

// ------------------------------------------------------------------ tokenise
// Split on anything that is not a letter or a digit, and keep digits together
// so that "12" and "345" are separate tokens rather than a stream of digits.
//
// INLINE, not a plain definition: this lives in a header and is included by more
// than one translation unit. Without the keyword every TU emitted its own copy
// and the link failed with LNK2005 (duplicate symbol).
inline std::vector<std::string> lex_tokens(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (unsigned char c : s) {
    if (std::isalnum(c)) {
      cur.push_back((char)std::tolower(c));
    } else {
      if (!cur.empty()) { out.push_back(cur); cur.clear(); }
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

// ---------------------------------------------------------------------------
//  Hybrid encoding: what to combine and why.
//
//  MEASURED, on identical probes (am-sim):
//      representation        margin    cross-intent inversions
//      character n-gram      0.0981    84 / 90      (no discrimination)
//      learned lexical       0.4890    21 / 90      (clear discrimination)
//
//  But the lexical model has a weakness that matters for this system: it throws
//  away SURFACE FORM. "C:/work/a.txt" and "C:/work/b.txt" tokenise to nearly the
//  same bag, so the embedding scores them almost identically -- and the memory
//  must distinguish exactly those two to retrieve the right episode.
//
//  The n-gram encoder has the opposite profile: no semantic structure, but exact
//  sensitivity to spelling, which is what protects path and expression retrieval.
//
//  So the memory is keyed on the CONCATENATION, each part L2 normalised and
//  scaled. n-gram keeps the precision, lexical supplies the meaning.
//
//      key = [ W_GRAM * gram(2048) | W_LEX * lex(160) ]
//
//  Weights are set so the two parts contribute comparably at their measured
//  typical magnitudes rather than by intuition: gram cosines run ~0.1-0.6 on
//  related text, lexical cosines ~0.4-0.9, so the lexical part is scaled down.
// ---------------------------------------------------------------------------
// 2048 + 160 = 2208, and 2208 / 32 = 69 exactly, so the ternary popcount path
// (which packs 32 dimensions per word) works on hybrid keys without padding.
inline constexpr int  HYB_GRAM_DIM = 2048;
inline constexpr int  HYB_LEX_DIM  = 160;
inline constexpr int  HYB_DIM      = HYB_GRAM_DIM + HYB_LEX_DIM;   // 2208
// Runtime-adjustable so the contribution of semantics can be ablated without
// a rebuild. A measured tradeoff deserves a measured knob.
extern float HYB_W_GRAM;
extern float HYB_W_LEX;

// Both parts are L2 normalised before scaling, so the concatenation is a
// weighted sum of two unit vectors and the key norm is sqrt(wg^2 + wl^2).
// Cosine between two hybrid keys therefore decomposes into
//     (wg^2 * gram_cos + wl^2 * lex_cos) / (wg^2 + wl^2)
// which is why the lexical weight is set from measured magnitudes rather than
// guessed: gram cosines on related text run ~0.1-0.6, lexical ~0.4-0.9.

// Concatenate a gram vector and a lexical vector into one key.
void hybrid_key(const std::vector<float>& gram, const std::vector<float>& lex,
                std::vector<float>& out);

// ------------------------------------------------------------- the model
struct LexModel {
  std::unordered_map<std::string, int> w2i;
  std::vector<std::string> words;
  std::vector<float> emb;      // |V| x DIM, row-major
  int DIM = 160;

  int id(const std::string& w) const {
    auto it = w2i.find(w);
    return it == w2i.end() ? -1 : it->second;
  }

  // Train on a corpus of sentences.
  void train(const std::vector<std::string>& corpus, int dim);

  // Mean of the word vectors, L2 normalised. Unknown words are skipped.
  void encode(const std::string& text, std::vector<float>& out) const;

  // Fraction of a sentence's tokens that were known. Low coverage means the
  // embedding has nothing to say and the caller should fall back.
  float coverage(const std::string& text) const;

  // ---- persistence ----
  // The embedding has to travel with the memory, otherwise a query encoded at
  // load time would land in a different space than the stored keys and every
  // retrieval would be noise. Plain little-endian, same convention as the
  // checkpoint files.
  bool save(std::ostream& f) const;
  bool load(std::istream& f);
};

} // namespace canticle

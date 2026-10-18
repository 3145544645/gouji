// ============================================================================
//  Hybrid key assembly.
//
//  The memory is one flat matrix, so a hybrid key has to be one vector. The two
//  parts are scaled and concatenated rather than mixed, which keeps the geometry
//  interpretable: cosine between keys decomposes into a weighted sum of a
//  surface-similarity term and a meaning-similarity term.
//
//  Order matters when debugging: gram first, lexical second, so a dump shows the
//  familiar 2048-wide prefix.
// ============================================================================
#include "canticle/lex.h"
#include <cmath>

namespace canticle {

// DEFAULT IS ZERO, and that is a measured decision, not a placeholder.
//
// The lexical embedding was integrated, then evaluated end-to-end (am-e2e):
//
//     configuration            novel synonyms   controls   held-out accuracy
//     structure only                  1/6          2/2          100.00%
//     hybrid (w_lex = 0.60)           1/6          2/2           94.59%
//
// It changed no behaviour on unseen synonyms while costing 5.4 points of
// held-out accuracy: the semantic part dilutes the surface precision that path
// and expression retrieval depend on, and for vocabulary absent from the corpus
// it contributes a near-random direction that competes with a good signal.
//
// The code is kept, and the weight is exposed, so the tradeoff can be re-measured
// rather than re-argued. Raise it only with a corpus large enough that unseen
// synonyms are in-vocabulary.
float HYB_W_GRAM = 1.00f;
float HYB_W_LEX  = 0.00f;

void hybrid_key(const std::vector<float>& gram, const std::vector<float>& lex,
                std::vector<float>& out) {
  // When the lexical weight is zero the key is just the gram vector, and its
  // DIMENSION must be HYB_GRAM_DIM, not HYB_DIM.
  //
  // The first version always emitted 2208 values. AmMem sizes itself to
  // HYB_GRAM_DIM (2048), so every append failed the dimension check and online
  // learning could never store anything -- with an error message that pointed at
  // the caller rather than at this function.
  if (HYB_W_LEX <= 0.f) {
    out.assign(gram.size(), 0.f);
    double gs = 0;
    for (float x : gram) gs += (double)x * x;
    const double gn = (gs > 1e-12) ? std::sqrt(gs) : 1.0;
    for (size_t i = 0; i < gram.size(); ++i)
      out[i] = (float)(gram[i] / gn) * HYB_W_GRAM;
    return;
  }

  out.assign(HYB_DIM, 0.f);

  // Each part is already L2 normalised by its encoder; normalise again here so
  // that a caller passing an unnormalised vector cannot silently dominate.
  auto norm = [](const std::vector<float>& v) {
    double s = 0;
    for (float x : v) s += (double)x * x;
    return (s > 1e-12) ? std::sqrt(s) : 1.0;
  };
  const double gn = norm(gram), ln = norm(lex);

  const int g = (int)std::min(gram.size(), (size_t)HYB_GRAM_DIM);
  for (int i = 0; i < g; ++i) out[i] = (float)(gram[i] / gn) * HYB_W_GRAM;

  const int l = (int)std::min(lex.size(), (size_t)HYB_LEX_DIM);
  for (int i = 0; i < l; ++i) out[HYB_GRAM_DIM + i] = (float)(lex[i] / ln) * HYB_W_LEX;
}

} // namespace canticle

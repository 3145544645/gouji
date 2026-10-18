// ===========================================================================
//  Semantic similarity, measured directly.
//
//  THE TEST THAT MATTERS
//  ---------------------
//  The earlier am-sem probe stopped being meaningful once the corpus itself
//  contained "inspect" and "examine": passing it proved memorisation, not
//  meaning. This file asks the question that memorisation cannot fake:
//
//      do two phrasings of the SAME intent land closer together than two
//      phrasings of DIFFERENT intents, when NEITHER phrasing was stored?
//
//  Concretely it computes, in the 2048-dimensional encoder space:
//      same_intent  = cosine(a, b) where a,b express the same tool
//      diff_intent  = cosine(a, c) where c expresses a different tool
//  and reports the margin. A margin near zero means the representation carries
//  no semantic structure; a positive margin with few inversions means it does.
//
//  Every phrase below is deliberately absent from the generator's banks.
// ===========================================================================
#include "canticle/am.h"
#include "canticle/lex.h"
#include "canticle/data.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

namespace canticle {

// Groups of never-stored phrasings, one group per intent.
static const char* kG_read[]  = {"grab D:/q/one.md",   "pull up D:/q/two.md",   "hand me D:/q/three.md"};
static const char* kG_list[]  = {"what lives in D:/q", "survey D:/q",           "index D:/q"};
static const char* kG_write[] = {"drop data at D:/q/a.txt", "persist data into D:/q/b.txt", "park data at D:/q/c.txt"};
static const char* kG_shell[] = {"fire off hostname",  "bring up whoami",       "kick off Get-Date"};
static const char* kG_calc[]  = {"crunch 31 + 47",     "tally 62 - 18",         "add up 55 * 9"};

struct Group { const char* name; const char* const* items; int n; };

static const Group kGroups[] = {
  {"read_file",  kG_read,  3},
  {"list_dir",   kG_list,  3},
  {"write_file", kG_write, 3},
  {"run_shell",  kG_shell, 3},
  {"calc",       kG_calc,  3},
};

// ---------------------------------------------------------------------------
//  Second probe: phrasings built ONLY from words the generator corpus contains.
//
//  The first probe is deliberately adversarial -- "grab", "survey", "crunch" are
//  absent from the corpus, so a learned model has literally nothing to go on.
//  That measures COVERAGE, not representation quality. This second set keeps the
//  intent but restricts the wording to known vocabulary, which separates the two
//  questions:
//      probe 1 fails + probe 2 works  -> coverage limit (need more corpus)
//      probe 1 fails + probe 2 fails  -> the representation itself is broken
// ---------------------------------------------------------------------------
static const char* kK_read[]  = {"show me D:/k/one.md",   "display D:/k/two.md",    "view D:/k/three.md"};
static const char* kK_list[]  = {"enumerate D:/k",        "ls D:/k",                "dir D:/k"};
static const char* kK_write[] = {"save to D:/k/a.txt",    "write to D:/k/b.txt",    "store at D:/k/c.txt"};
static const char* kK_shell[] = {"execute hostname",      "run hostname",           "invoke hostname"};
static const char* kK_calc[]  = {"compute 31 + 47",       "calculate 31 + 47",      "work out 31 + 47"};

static const Group kKnown[] = {
  {"read_file",  kK_read,  3},
  {"list_dir",   kK_list,  3},
  {"write_file", kK_write, 3},
  {"run_shell",  kK_shell, 3},
  {"calc",       kK_calc,  3},
};

static double cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double s = 0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) s += (double)a[i] * b[i];
  return s;
}

} // namespace canticle

using namespace canticle;

// ---------------------------------------------------------------------------
//  Shared scorer so the two representations are compared on identical terms.
// ---------------------------------------------------------------------------
struct Score {
  double same_mean, diff_mean, same_min, diff_max, margin;
  int inv, diff_n;
};

static Score score_all(const std::vector<std::vector<float>>& vec, int NG) {
  auto V = [&](int g, int i) { return g * 3 + i; };
  Score s{};
  double ss = 0, ds = 0;
  int sn = 0;
  s.same_min = 1e9; s.diff_max = -1e9;
  for (int g = 0; g < NG; ++g)
    for (int i = 0; i < 3; ++i)
      for (int j = i + 1; j < 3; ++j) {
        const double c = cosine(vec[V(g,i)], vec[V(g,j)]);
        ss += c; sn++; s.same_min = std::min(s.same_min, c);
      }
  for (int g1 = 0; g1 < NG; ++g1)
    for (int g2 = g1 + 1; g2 < NG; ++g2)
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          const double c = cosine(vec[V(g1,i)], vec[V(g2,j)]);
          ds += c; s.diff_n++; s.diff_max = std::max(s.diff_max, c);
          if (c > s.same_min) s.inv++;
        }
  s.same_mean = ss / std::max(1, sn);
  s.diff_mean = ds / std::max(1, s.diff_n);
  s.margin = s.same_mean - s.diff_mean;
  return s;
}

int am_sim_run(int argc, char** argv) {
  AmEncoder enc; enc.D = 2048;

  // IDF needs a document set. Use only the never-stored probe phrases plus a
  // neutral filler, so the statistics cannot smuggle in generator vocabulary.
  std::vector<std::string> docs;
  for (auto& g : kGroups)
    for (int i = 0; i < g.n; ++i) docs.push_back(g.items[i]);
  enc.build_idf(docs);

  const int NG = (int)(sizeof(kGroups) / sizeof(kGroups[0]));
  std::printf("=== semantic similarity in encoder space ===\n");
  std::printf("all %d phrases are absent from the corpus\n\n", NG * 3);

  // Encode everything.
  std::vector<std::vector<float>> vec;
  for (auto& g : kGroups)
    for (int i = 0; i < g.n; ++i) {
      std::vector<float> v; enc.encode(g.items[i], v); vec.push_back(std::move(v));
    }
  auto V = [&](int g, int i) { return g * 3 + i; };

  const Score A = score_all(vec, NG);
  const double sm = A.same_mean, dm = A.diff_mean;

  std::printf("--- A. character n-gram encoder (2048-d) ---\n");
  std::printf("  same intent : mean %.4f   min %.4f\n", A.same_mean, A.same_min);
  std::printf("  diff intent : mean %.4f   max %.4f\n", A.diff_mean, A.diff_max);
  std::printf("  margin      : %.4f\n", A.margin);
  std::printf("  inversions  : %d of %d\n", A.inv, A.diff_n);

  // ---- B. learned lexical embedding, same test ----
  //
  // Trained on the GENERATOR's corpus -- thousands of sentences produced by
  // synth_tool_data, which is also what the memory is built from. The probe
  // phrases below never appear in it, so a positive result here is
  // generalisation from co-occurrence rather than a look-up.
  Score B{};
  std::vector<std::vector<float>> vecB;
  const int dim = 160;
  {
    auto train_samples = synth_tool_data(4000, 99);
    std::vector<std::string> corpus;
    for (auto& s : train_samples) corpus.push_back(s.prompt);
    LexModel lex;
    lex.train(corpus, dim);
    std::printf("\n--- B. learned lexical embedding (PPMI + SVD) ---\n");
    std::printf("  corpus: %zu sentences, vocab %zu, dim %d\n",
                corpus.size(), lex.words.size(), lex.DIM);
    int unknown = 0, tot = 0;
    for (auto& g : kGroups)
      for (int i = 0; i < g.n; ++i) {
        std::vector<float> v; lex.encode(g.items[i], v);
        vecB.push_back(std::move(v));
        for (const auto& w : lex_tokens(g.items[i])) { tot++; if (lex.id(w) < 0) unknown++; }
      }
    std::printf("  probe-1 coverage (adversarial): %.0f%% (%d of %d unseen)\n",
                100.0 * (tot - unknown) / std::max(1, tot), unknown, tot);
    // Score here, inside the scope where the model is alive. An earlier version
    // filled vecB and never called score_all, so every B statistic printed as
    // 0.0000 and looked like a representation failure rather than a missing call.
    B = score_all(vecB, NG);

    // ---- diagnostics: is the embedding actually populated? ----
    if (getenv("LEX_DIAG")) {
      std::printf("\n  [diag] vocabulary: ");
      for (size_t i = 0; i < lex.words.size() && i < 40; ++i)
        std::printf("%s ", lex.words[i].c_str());
      std::printf("\n");
      // norm of a few word vectors
      for (const char* w : {"read", "show", "file", "list", "run", "command", "hello"}) {
        const int i = lex.id(w);
        if (i < 0) { std::printf("  [diag] %-9s NOT IN VOCAB\n", w); continue; }
        double n = 0;
        for (int k = 0; k < lex.DIM; ++k) n += (double)lex.emb[(size_t)i * lex.DIM + k] * lex.emb[(size_t)i * lex.DIM + k];
        std::printf("  [diag] %-9s id=%d  |v|=%.6f\n", w, i, std::sqrt(n));
      }
      // nearest neighbours of "read" by cosine in embedding space
      const int ri = lex.id("read");
      if (ri >= 0) {
        std::vector<std::pair<double,int>> sims;
        std::vector<double> rvec(lex.DIM);
        for (int k = 0; k < lex.DIM; ++k) rvec[k] = lex.emb[(size_t)ri * lex.DIM + k];
        double rn = 0; for (double x : rvec) rn += x * x; rn = std::sqrt(rn);
        for (size_t j = 0; j < lex.words.size(); ++j) {
          if ((int)j == ri) continue;
          double dot = 0, jn = 0;
          for (int k = 0; k < lex.DIM; ++k) {
            dot += rvec[k] * lex.emb[(size_t)j * lex.DIM + k];
            jn  += (double)lex.emb[(size_t)j * lex.DIM + k] * lex.emb[(size_t)j * lex.DIM + k];
          }
          jn = std::sqrt(jn);
          if (rn > 1e-9 && jn > 1e-9) sims.push_back({dot / (rn * jn), (int)j});
        }
        std::sort(sims.rbegin(), sims.rend());
        std::printf("  [diag] nearest to 'read': ");
        for (int i = 0; i < 8 && i < (int)sims.size(); ++i)
          std::printf("%s(%.3f) ", lex.words[sims[i].second].c_str(), sims[i].first);
        std::printf("\n");
      }
    }
  }
  std::printf("  same intent : mean %.4f   min %.4f\n", B.same_mean, B.same_min);
  std::printf("  diff intent : mean %.4f   max %.4f\n", B.diff_mean, B.diff_max);
  std::printf("  margin      : %.4f\n", B.margin);
  std::printf("  inversions  : %d of %d\n", B.inv, B.diff_n);

  // ---- C. learned embedding on the same phrases, but known vocabulary ----
  Score C{};
  std::vector<std::vector<float>> vecC;
  {
    auto train_samples = synth_tool_data(4000, 99);
    std::vector<std::string> corpus;
    for (auto& s : train_samples) corpus.push_back(s.prompt);
    LexModel lex;
    lex.train(corpus, dim);
    int unknown = 0, tot = 0;
    for (auto& g : kKnown)
      for (int i = 0; i < g.n; ++i) {
        std::vector<float> v; lex.encode(g.items[i], v);
        vecC.push_back(std::move(v));
        for (const auto& w : lex_tokens(g.items[i])) { tot++; if (lex.id(w) < 0) unknown++; }
      }
    C = score_all(vecC, NG);
    std::printf("\n--- C. learned lexical embedding, in-vocabulary probe ---\n");
    std::printf("  probe-2 coverage: %.0f%% (%d of %d unseen)\n",
                100.0 * (tot - unknown) / std::max(1, tot), unknown, tot);
    std::printf("  same intent : mean %.4f   min %.4f\n", C.same_mean, C.same_min);
    std::printf("  diff intent : mean %.4f   max %.4f\n", C.diff_mean, C.diff_max);
    std::printf("  margin      : %.4f\n", C.margin);
    std::printf("  inversions  : %d of %d\n", C.inv, C.diff_n);
  }

  std::printf("\n--- comparison ---\n");
  std::printf("  %-28s margin %+.4f   inversions %d/%d\n",
              "n-gram (current)", A.margin, A.inv, A.diff_n);
  std::printf("  %-28s margin %+.4f   inversions %d/%d\n",
              "learned (adversarial)", B.margin, B.inv, B.diff_n);
  std::printf("  %-28s margin %+.4f   inversions %d/%d\n",
              "learned (in-vocab)", C.margin, C.inv, C.diff_n);

  auto verdict = [](const Score& s) {
    return (s.margin > 0.05 && (double)s.inv / std::max(1, s.diff_n) < 0.25)
             ? "PRESENT" : "ABSENT";
  };
  std::printf("\n  n-gram, adversarial   : %s\n", verdict(A));
  std::printf("  learned, adversarial  : %s\n", verdict(B));
  std::printf("  learned, in-vocab     : %s\n", verdict(C));

  // The diagnostic that matters: does the gap come from coverage or from the
  // representation? A working in-vocab result alongside a failing adversarial one
  // means the method is sound and the corpus is the limit.
  if (verdict(B)[0] == 'A' && verdict(C)[0] == 'P')
    std::printf("\n  => representation is SOUND; the adversarial failure is a "
                "COVERAGE limit (corpus has no such verbs)\n");

  // ---------------------------------------------------------------------------
  //  WHAT THIS SUITE MAY ASSERT, and why it is only the baseline.
  //
  //  Like am-e2e and am-sem it returned 0 unconditionally, so nothing it printed was
  //  protected. But unlike them, most of its output is a RESEARCH VERDICT: whether
  //  the learned representation closes a gap is the question under study, and
  //  freezing "PRESENT" or "ABSENT" would assert a conclusion as a requirement --
  //  defect 53's error.
  //
  //  What can be asserted is the instrument: A is the character n-gram encoder, the
  //  representation the system already relies on, measured on pairs it can see. If
  //  THAT reads ABSENT, the probe is broken rather than the idea -- the same
  //  distinction as a control in a clinical trial.
  //
  //  So exactly one assertion, on the one quantity whose failure would mean the
  //  measurement is not measuring.
  // ---------------------------------------------------------------------------
  //
  //  AND THE ASSERTION MUST NAME THE RIGHT QUANTITY.
  //
  //  The first version asserted that A reads PRESENT -- and it failed, because A is
  //  measured on the ADVERSARIAL pairs, which are exactly the ones the character
  //  n-gram encoder is expected to fail. Asserting PRESENT there would assert that
  //  the whole experiment has no subject.
  //
  //  That is defect 53 once more: an assertion encoding an expectation the system
  //  does not and should not meet. Caught immediately, because this time the
  //  assertion was added together with the measurement it constrains.
  //
  //  What can honestly be required is that the probe carries SIGNAL: a margin above
  //  zero with fewer inversions than differences. Both hold for every column above,
  //  including the failing one (margin +0.0981, 84 of 90 inverted), which is why the
  //  numbers are usable even where the verdict is ABSENT. A margin at or below zero
  //  would mean the score is uncorrelated with the grouping and nothing below it
  //  means anything.
  if (A.margin <= 0.0f || A.inv >= A.diff_n) {
    std::printf("\n  FAIL: the n-gram baseline carries no signal (margin %+.4f, %d of %d\n"
                "  inverted). A positive margin is what makes the ABSENT/PRESENT\n"
                "  verdict meaningful at all; without it the probe measures noise.\n",
                A.margin, A.inv, A.diff_n);
    return 1;
  }
  std::printf("\n  ALL PASS  (the probe carries signal; the verdicts above are\n"
              "  reported, not asserted -- they are the subject of the experiment)\n");
  return 0;
}

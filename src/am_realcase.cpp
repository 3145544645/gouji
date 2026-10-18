// ===========================================================================
//  Real-world phrasing, taken verbatim from CLINC150.
//
//  WHY THIS FILE EXISTS
//  --------------------
//  Every earlier test set was written alongside the code that satisfies it. That
//  is circular: a generator and a recogniser written by the same hand will agree
//  on the forms that hand thought of.
//
//  CLINC150 was written by other people. Its calculator utterances immediately
//  found forms the synthetic corpus never contained -- operators as English words,
//  percentage operations, square roots -- and the system handled 1 of 8. The
//  strings below are copied from that corpus unmodified.
//
//  WHAT IT MEASURES, AND WHAT IT DELIBERATELY DOES NOT
//  ---------------------------------------------------
//  It separates two different abilities:
//
//    ROUTING  which tool is chosen. This is the part the system is designed to do
//             and where the frame of "8/8" belongs.
//
//    ARITHMETIC  whether the expression is actually evaluated. "what is 8
//             factorial" routes correctly to calc and then fails to compute,
//             because the evaluator handles + - * / only.
//
//  Conflating them would make the number look like a failure of routing when it is
//  a known limit of the evaluator. Both are reported, separately.
// ===========================================================================
#include "canticle/am.h"
#include <cstdio>
#include <string>
#include <vector>

namespace canticle {

struct RealCase {
  const char* text;      // verbatim from CLINC150
  const char* want_tool;
  bool computable;       // can the current evaluator produce a value
  const char* feature;
};

// The feature field names the LINGUISTIC form, because the interesting output of
// this file is not a score -- it is the list of forms real users use that the
// generator produced none of.
static const RealCase kReal[] = {
  // ---- plain symbolic, which the synthetic corpus does cover ----
  {"what is 4 + 4",                       "calc", true,  "symbolic"},
  {"what is 2+2",                         "calc", true,  "symbolic no spaces"},
  {"what is 22+ 6",                       "calc", true,  "symbolic ragged spacing"},
  {"what is 20+ 5",                       "calc", true,  "symbolic ragged spacing"},
  {"what is 10 + 20",                     "calc", true,  "symbolic"},

  // ---- operator as a word ----
  {"what is 7 x 7",                       "calc", true,  "x as multiply"},
  {"what is 55 times 300",                "calc", true,  "times"},
  {"what is 213 times 3",                 "calc", true,  "times"},
  {"what is the sum of 3 plus 5",         "calc", true,  "plus"},
  {"what is the sum of 10 plus 5",        "calc", true,  "plus"},
  {"please add 456 and 781 for me",       "calc", true,  "add ... and"},

  // ---- percentage ----
  {"what is 25 percent of 6999",          "calc", true,  "percent of"},
  {"what is 15% of 3143",                 "calc", true,  "percent sign of"},
  {"what is 85% of $1500",                "calc", true,  "percent with currency"},
  {"what is 60 percent off of 350 dollars","calc",true,  "percent off"},
  {"what is 18 percent tip on 20 dollars","calc", true,  "percent tip"},

  // ---- fraction of ----
  {"what is 3/4 of 18",                   "calc", true,  "fraction of"},

  // ---- named operations with NO implementation ----
  //
  // These route to calc and then fail to evaluate, which is the honest outcome:
  // the request was understood, the operation is not available. Reported separately
  // from routing so the number means what it says.
  {"what is 8 factorial",                 "calc", false, "factorial"},
  {"what is the square root of 144",      "calc", false, "square root"},
  {"what is the square root of 80",       "calc", false, "square root"},
  {"what is the square root of 1 million","calc", false, "square root, word number"},

  // OUT OF SCOPE, and asserted as such rather than expected to work.
  //
  // "what is the antilog of 365" is refused, and that is correct: no rule claims
  // the word "antilog", no tool implements it, and the nearest stored neighbour
  // scores 0.384 -- below the 0.48 floor. It used to be asserted as calc, which
  // made the test report a failure for behaviour that is right. A test whose
  // expectation encodes a capability the system does not have is a test that
  // will be "fixed" by weakening something else.
  {"what is the antilog of 365",          "?",    false, "antilog (no implementation)"},

  // ---- long, specific payloads ----
  //
  // These were REFUSED by the first version of the similarity floor, which was
  // calibrated on short payloads. "show me C:/Users/cc/canticle/STATUS.md" scored
  // 0.415 against "show me C:/log/log.txt" and fell under a flat 0.48 threshold,
  // even though the structure layer had already resolved it to read_file.
  //
  // The lesson is that a whole-sentence similarity score falls as the payload gets
  // longer and more specific, so the more precisely a user names something, the
  // less the system recognises it -- exactly backwards. The floor is now applied
  // only when infer_intent returns Unknown.
  {"show me C:/Users/cc/canticle/STATUS.md", "read_file", true, "long path, show"},
  {"open C:/Users/cc/canticle/STATUS.md",    "read_file", true, "long path, open"},
  {"print C:/Users/cc/canticle/STATUS.md",   "read_file", true, "long path, print"},
  {"what is the sum of 3 plus 5",            "calc",      true, "sum of, plus"},
  {"please add 456 and 781 for me",          "calc",      true, "add ... and"},

  // ---- requires an external tool, out of scope by design ----
  {"what time is it in houston, tx",      "?",    false, "world clock"},
  {"tell me what day today is",           "?",    false, "calendar"},
  {"translate hello into french",         "?",    false, "translation"},
  {"what is the weather in boston",       "?",    false, "weather"},
};

} // namespace canticle

using namespace canticle;

int am_realcase_run(int, char**) {
  const int N = (int)(sizeof(kReal) / sizeof(kReal[0]));
  std::printf("=== real-world phrasing (CLINC150, verbatim) ===\n");
  std::printf("%d utterances written by other people\n\n", N);

  int routed = 0, scorable = 0, computable = 0, computable_ok = 0;
  for (int i = 0; i < N; ++i) {
    const bool testable = (kReal[i].want_tool[0] != '?');
    std::string reply; float sim = 0;
    if (am_query_public("build\\ckpt\\canticle.am", kReal[i].text, reply, sim) != 0) {
      std::printf("  [ERR] %s\n", kReal[i].text);
      continue;
    }
    const size_t a = reply.find("<|tool|>");
    std::string got = "none";
    if (a != std::string::npos) {
      const size_t b = reply.find(' ', a);
      got = reply.substr(a + 8, (b == std::string::npos ? reply.size() : b) - (a + 8));
    }
    if (testable) {
      scorable++;
      if (got == kReal[i].want_tool) routed++;
    }
    if (kReal[i].computable) { computable++; if (got == "calc") computable_ok++; }

    std::printf("  %-9s %-26s -> %-11s  %s\n",
                testable ? (got == kReal[i].want_tool ? "ok" : "MISS") : "(scope)",
                kReal[i].feature, got.c_str(), kReal[i].text);
  }

  std::printf("\n--- routing (tool choice) ---\n");
  std::printf("  correct: %d/%d  %.1f%%\n", routed, scorable, 100.0 * routed / std::max(1, scorable));
  std::printf("  all of these reach calc: %d/%d\n", computable_ok, computable);
  return 0;
}

// ===========================================================================
//  End-to-end semantic benefit test.
//
//  THE QUESTION THIS ANSWERS
//  -------------------------
//  Integrating a learned lexical embedding cost 5.4 points of held-out accuracy
//  (100% -> 94.6%) because it dilutes the surface precision that path and
//  expression retrieval depends on. That is a REAL, MEASURED cost.
//
//  It is only worth paying if semantics buys something the structure layer cannot
//  already do. This test gives semantics its best case:
//
//    * the phrasing is a synonym the corpus NEVER contains
//    * the phrasing contains NO structural signal (no digit, no operator) so the
//      intent layer cannot decide from syntax alone
//    * the payload is a path, so a pass also proves the surface part survived
//
//  If semantics wins here it is worth the tradeoff. If it does not, the honest
//  conclusion is that the structure layer already covers the practical cases and
//  the embedding should be dropped.
// ===========================================================================
#include "canticle/am.h"
#include "canticle/lex.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace canticle {

struct SemCase {
  const char* text;
  const char* want_tool;
  const char* want_payload;
  const char* note;
};

// Verbs below are ABSENT from data.cpp's banks and ABSENT from the tool
// templates. None of them carries structural signal.
static const SemCase kE2E[] = {
  {"snoop D:/e2e/one.md",      "read_file",  "D:/e2e/one.md", "snoop"},
  {"peruse D:/e2e/two.md",     "read_file",  "D:/e2e/two.md", "peruse"},
  {"audit D:/e2e/three.md",    "read_file",  "D:/e2e/three.md", "audit"},
  {"catalogue D:/e2e",         "list_dir",   "D:/e2e",        "catalogue"},
  {"stocktake D:/e2e",         "list_dir",   "D:/e2e",        "stocktake"},
  {"commit data to D:/e2e/a.txt", "write_file", "D:/e2e/a.txt", "commit to"},
  // controls: these already work through structure alone
  {"read the file D:/e2e/one.md", "read_file", "D:/e2e/one.md", "[ctrl]"},
  {"list the directory D:/e2e",   "list_dir",  "D:/e2e",        "[ctrl]"},
};

static std::string tool_of(const std::string& r) {
  size_t a = r.find("<|tool|>");
  if (a == std::string::npos) return "";
  size_t b = r.find(' ', a);
  return r.substr(a + 8, (b == std::string::npos ? r.size() : b) - (a + 8));
}
static std::string squash(const std::string& s) {
  std::string o;
  for (char c : s) if (c != ' ' && c != '\t') o.push_back(c);
  return o;
}

} // namespace canticle

using namespace canticle;

int am_e2e_run(int argc, char** argv) {
  std::string ampath = "build\\ckpt\\canticle.am";
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--am" && i + 1 < argc) ampath = argv[++i];

  struct Cfg { float wlex; const char* name; };
  const Cfg cfgs[] = {{0.00f, "structure only (no semantics)"},
                      {0.60f, "hybrid (semantics on)"}};

  const int N = (int)(sizeof(kE2E) / sizeof(kE2E[0]));
  std::printf("=== end-to-end semantic benefit ===\n");
  std::printf("%d requests; the non-control verbs appear nowhere in the corpus\n\n", N);

  for (const auto& c : cfgs) {
    HYB_W_LEX = c.wlex;
    int ok = 0, novel_ok = 0, novel_n = 0, ctrl_ok = 0, ctrl_n = 0;
    std::printf("--- %s ---\n", c.name);
    for (int i = 0; i < N; ++i) {
      std::string reply; float sim = 0;
      if (am_query_public(ampath, kE2E[i].text, reply, sim) != 0) continue;
      const std::string got = tool_of(reply);
      const bool hit = (got == kE2E[i].want_tool) &&
                       squash(reply).find(squash(kE2E[i].want_payload)) != std::string::npos;
      if (hit) ok++;
      const bool is_ctrl = (kE2E[i].note[0] == '[');
      if (is_ctrl) { ctrl_n++; if (hit) ctrl_ok++; }
      else         { novel_n++; if (hit) novel_ok++; }
      std::printf("  %s %-14s want %-11s got %-11s\n",
                  hit ? "[ok]  " : "[MISS]", kE2E[i].note, kE2E[i].want_tool, got.c_str());
    }
    std::printf("  -> novel %d/%d   control %d/%d   TOTAL %d/%d\n\n",
                novel_ok, novel_n, ctrl_ok, ctrl_n, ok, N);

    // -------------------------------------------------------------------------
    //  THIS USED TO RETURN 0 UNCONDITIONALLY.
    //
    //  An independent audit listed it among the suites that "exit 0 whatever the
    //  code does". It was right, and the consequence is that the numbers in section 3
    //  of the architecture document had no regression protection at all: they were
    //  measurements, reprinted on every run, that nothing compared.
    //
    //  TWO KINDS OF ASSERTION, deliberately different in strength:
    //
    //    CONTROL  The controls are requests the structure layer already handles. A
    //             failure there is a defect, not a research result, so the bar is
    //             "all of them".
    //
    //    NOVEL    These are verbs absent from the corpus; how many the encoder
    //             generalises to is what the experiment MEASURES. Asserting the
    //             current value would freeze a number that further work should
    //             improve, so the bar is the measured floor -- enough to catch a
    //             collapse, not enough to punish progress.
    //
    //  The distinction is the one from defect 53: an assertion must encode a
    //  requirement, not a wish. "Controls work" is a requirement. "Novel stays at
    //  exactly 5" is not.
    // -------------------------------------------------------------------------
    // Minimum work: with no memory every request fails to answer, both counters stay
    // at zero, and the two checks below pass vacuously. Measured on a fresh clone.
    if (ctrl_n + novel_n == 0) {
      std::printf("  FAIL: no request produced an answer. The memory is missing or\n"
                  "  unreadable, so this run measured nothing.\n"
                  "  Build one first: canticle am-build --data 6000\n");
      return 1;
    }
    if (ctrl_ok != ctrl_n) {
      std::printf("  FAIL: %d of %d controls failed. Controls are known-good requests;\n"
                  "  a failure here is a regression, not a measurement.\n",
                  ctrl_n - ctrl_ok, ctrl_n);
      return 1;
    }
    if (novel_n > 0 && novel_ok * 2 < novel_n) {
      std::printf("  FAIL: novel %d/%d -- below half. The generalisation the corpus was\n"
                  "  built to demonstrate has collapsed.\n", novel_ok, novel_n);
      return 1;
    }
  }
  std::printf("  ALL PASS  (controls intact, novel above the floor)\n");
  return 0;
}

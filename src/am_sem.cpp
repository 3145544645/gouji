
// ===========================================================================
//  Semantic robustness: the layer above gaps A-D.
//
//  Every gap so far was closed with STRUCTURE. infer_intent reads digits,
//  operators, paths and a list of English verbs; extract_payload pulls the tail
//  token. None of that understands meaning.
//
//  This file asks what structure cannot answer: does the system recognise a
//  REQUEST it has never seen, phrased in words it has never stored?
//
//  Test set design:
//    * every payload is a path the memory has never held, so a pass cannot come
//      from having memorised the answer
//    * the verbs share almost no character n-grams with the stored templates
//      ("inspect" vs "read the file")
//    * a control group of known-good phrasings is included so that a regression
//      in existing behaviour shows up in the same run
// ===========================================================================
#include "canticle/am.h"
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

static const SemCase kSem[] = {
  // ---- never-stored verbs, never-stored paths ----
  {"inspect D:/repo/zeta.md",                 "read_file",  "D:/repo/zeta.md", "inspect"},
  {"examine D:/repo/zeta.md",                 "read_file",  "D:/repo/zeta.md", "examine"},
  {"peek at D:/repo/zeta.md",                 "read_file",  "D:/repo/zeta.md", "peek at"},
  {"give me the contents of D:/repo/zeta.md", "read_file",  "D:/repo/zeta.md", "verbose read"},
  {"save this into D:/repo/new.txt",          "write_file", "D:/repo/new.txt", "save into"},
  {"enumerate D:/repo",                       "list_dir",   "D:/repo",         "enumerate"},
  {"execute hostname",                        "run_shell",  "hostname",        "execute"},
  {"evaluate 45 * 12",                        "calc",       "45*12",           "evaluate"},
  // ---- control group: behaviour that already worked ----
  {"show me D:/repo/zeta.md",                 "read_file",  "D:/repo/zeta.md", "[ctrl] show me"},
  {"what is 45 * 12",                         "calc",       "45*12",           "[ctrl] what is"},
  {"read the file D:/repo/zeta.md",           "read_file",  "D:/repo/zeta.md", "[ctrl] read the file"},
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

int am_sem_run(int argc, char** argv) {
  std::string ampath = "build\\ckpt\\canticle.am";
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--am" && i + 1 < argc) ampath = argv[++i];

  const int N = (int)(sizeof(kSem) / sizeof(kSem[0]));
  std::printf("=== semantic robustness ===\n");
  std::printf("%d requests, none of them stored\n\n", N);

  int ok = 0, ctrl_ok = 0, ctrl_n = 0, novel_ok = 0, novel_n = 0;
  for (int i = 0; i < N; ++i) {
    std::string reply; float sim = 0;
    if (am_query_public(ampath, kSem[i].text, reply, sim) != 0) {
      std::printf("  [ERR]  %s\n", kSem[i].text);
      continue;
    }
    const std::string got = tool_of(reply);
    const bool tool_ok = (got == kSem[i].want_tool);
    const bool pay_ok = squash(reply).find(squash(kSem[i].want_payload)) != std::string::npos;
    const bool hit = tool_ok && pay_ok;
    if (hit) ok++;

    const bool is_ctrl = (kSem[i].note[0] == '[');
    if (is_ctrl) { ctrl_n++; if (hit) ctrl_ok++; }
    else         { novel_n++; if (hit) novel_ok++; }

    std::printf("  %s %-22s want %-11s got %-11s sim=%.3f\n",
                hit ? "[ok]  " : "[MISS]", kSem[i].note,
                kSem[i].want_tool, got.c_str(), sim);
    if (!hit) std::printf("          reply: %s\n", reply.c_str());
  }

  std::printf("\n--- breakdown ---\n");
  std::printf("  control (known-good) : %d/%d\n", ctrl_ok, ctrl_n);
  std::printf("  novel   (never seen) : %d/%d\n", novel_ok, novel_n);
  std::printf("  OVERALL              : %d/%d  %.1f%%\n", ok, N, 100.0 * ok / N);

  // Same two-tier assertion as am-e2e, and for the same reason: this suite returned
  // 0 unconditionally, so the "unseen synonym" figures quoted in the architecture
  //  document were reprinted every run and compared by nothing. An independent audit
  //  noticed the asymmetry with am-para, which does return a failure code.
  //
  //   control  known-good, so all must pass -- a failure is a defect.
  //   novel    the experiment's subject; asserted only against collapse.
  // ---------------------------------------------------------------------------
  //  MINIMUM WORK. This suite reported ALL PASS on a machine with no memory at all:
  //  every request printed [ERR], so ctrl_n and novel_n were both zero, and the two
  //  assertions below were skipped -- "0 of 0 controls failed" and "0 is not less
  //  than 0". Nothing was measured and the verdict was PASS.
  //
  //  That is defect 105's shape exactly, and it is the reason a suite must state how
  //  much work it expects: a mechanism that evaluated nothing and a mechanism that
  //  found nothing produce the same output.
  // ---------------------------------------------------------------------------
  if (ctrl_n + novel_n == 0) {
    std::printf("  FAIL: no request produced an answer. The memory is missing or\n"
                "  unreadable, so this run measured nothing.\n"
                "  Build one first: canticle am-build --data 6000\n");
    return 1;
  }
  if (ctrl_ok != ctrl_n) {
    std::printf("  FAIL: %d of %d controls failed.\n", ctrl_n - ctrl_ok, ctrl_n);
    return 1;
  }
  if (novel_n > 0 && novel_ok * 2 < novel_n) {
    std::printf("  FAIL: novel %d/%d -- below half, the generalisation has collapsed.\n",
                novel_ok, novel_n);
    return 1;
  }
  std::printf("  ALL PASS  (controls intact, novel above the floor)\n");
  return 0;
}

// ===========================================================================
//  Automatic precedence violation hunting.
//
//  THE PROBLEM THIS ADDRESSES
//  --------------------------
//  Defect 65: the first precedence table put CONTAINMENT above STRONG VERBS,
//  because " in <path>" names an operation while a verb is just a word. The
//  reasoning sounded right and the precedence test passed -- because that test
//  only asserted the conflicts its author thought of. The full regression found
//  the error: "put hello in C:/tmp/todo.json" became a directory listing, turning
//  a WRITE into a read.
//
//  A test whose coverage depends on the author's imagination cannot catch the
//  cases the author did not imagine. So this file does not test opinions: it
//  ENUMERATES every combination of two signals and reports which pairs are not
//  routed as the stronger signal alone would be.
//
//  HOW A PAIR IS BUILT
//  -------------------
//  The corpus generator already knows a prompt's intended tool. For each pair of
//  (promptA -> toolA, promptB -> toolB) the hunter forms a sentence that carries
//  BOTH signal kinds and asks: does infer_intent still return something sensible?
//
//  "Sensible" is defined against a single rule, stated once:
//
//      adding a second signal must not change the tool that the FIRST signal
//      already determines, UNLESS the second signal is strictly stronger.
//
//  Rather than encode that hierarchy by hand -- which is how defect 65 happened --
//  the hunter reports the OBSERVED table and flags every pair where combining
//  signals produces a tool neither signal alone would produce. Those are the
//  candidates, and they are enumerated rather than remembered.
// ===========================================================================
#include "canticle/am.h"
#include "canticle/data.h"
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

using namespace canticle;

static const char* iname(Intent i) { return intent_name(i); }

int am_hunt_run(int argc, char** argv) {
  int limit = 400;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--limit" && i + 1 < argc) limit = atoi(argv[++i]);

  const auto corpus = synth_tool_data(limit * 6, 4242);

  // ---- collect representative prompts per tool ----
  //
  // SAMPLING BIAS, and it produced a wrong baseline on the first run.
  //
  // The generator emits the newer tools only on every 7th, 11th and 13th
  // iteration, so taking the first N samples of the whole corpus collected ZERO
  // prompts for mkdir, search, stat and copy -- and the baseline then reported
  // "unknown" for all four, which looks like a routing failure and is actually an
  // empty bucket.
  //
  // Two fixes, both needed:
  //   * the corpus is larger, so every residue class is populated;
  //   * samples are selected by iterating the WHOLE corpus rather than stopping
  //     after the first few of each tool, so a sparse class still contributes.
  //
  // A baseline built from an empty sample is worse than no baseline: it reports a
  // confident wrong answer, which is the same failure mode as the bare-path
  // default and the fabricated tool calls in defect 40.
  (void)0;
  std::map<std::string, std::vector<std::string>> by_tool;
  for (const auto& s : corpus) {
    const size_t a = s.reply.find("<|tool|>");
    if (a == std::string::npos) continue;
    const size_t b = s.reply.find(' ', a);
    const std::string tool = s.reply.substr(a + 8,
        (b == std::string::npos ? s.reply.size() : b) - (a + 8));
    if (by_tool[tool].size() < 6) by_tool[tool].push_back(s.prompt);
  }
  // ---------------------------------------------------------------------------
  //  SPARSITY IS CHECKED AGAINST THE GENERATOR'S OWN TALLY, not by counting what
  //  this harness happened to collect.
  //
  //  The first version inferred the problem from "by_tool has no entry for this
  //  tool", which conflates two different facts: the corpus not containing a tool,
  //  and the harness not drawing enough samples to reach one. Those need different
  //  fixes, and the harness cannot tell them apart from its own bucket.
  //
  //  detail::last_tool_coverage() is the generator's own count, so the two cases
  //  become distinguishable and the message can name the right one.
  // ---------------------------------------------------------------------------
  {
    static const char* kExpected[] = {"read_file","list_dir","write_file","run_shell",
                                      "calc","stat","search","mkdir","copy"};
    const auto& cov = detail::last_tool_coverage();
    for (const char* t : kExpected) {
      const int have = cov.count(t) ? cov.at(t) : 0;
      const bool drew = by_tool.find(t) != by_tool.end();
      if (have == 0)
        std::printf("  [warn] the CORPUS contains no \"%s\" sample at all.\n", t);
      else if (!drew)
        std::printf("  [warn] corpus has %d \"%s\" samples but this harness drew none;\n"
                    "         raise --limit until it does, or the baseline is unmeasured.\n",
                    have, t);
      else if (have < 6)
        std::printf("  [note] only %d \"%s\" samples exist; results for it are thin.\n",
                    have, t);
    }
  }

  std::printf("=== automatic precedence violation hunt ===\n");
  std::printf("tools in corpus: ");
  for (const auto& kv : by_tool) std::printf("%s(%zu) ", kv.first.c_str(), kv.second.size());
  std::printf("\n\n");

  // ---- the single-signal baseline ----
  //
  // MEASURED AGAINST THE ROUTER, NOT against infer_intent alone.
  //
  // The first version called infer_intent directly and reported "unknown" for
  // mkdir, search, stat and copy -- every one of which routes correctly in
  // practice. That is not a bug in either component:
  //
  //     infer_intent   returns Unknown when the structure layer cannot decide.
  //                    A bare path is deliberately Unknown so the SESSION can
  //                    inherit the previous turn's tool (defect 15's guard).
  //     the router     takes Unknown and falls back to retrieval, which then picks
  //                    the right family from the embedding.
  //
  // So "what tool does this prompt produce" is a question about the ROUTER, and
  // asking infer_intent answers a different question -- one that happens to read
  // as a failure. The baseline now goes through the same path a real turn does.
  std::map<std::string, std::string> baseline;   // tool -> the tool actually produced
  std::printf("-- single-signal baseline (via the router) --\n");
  for (const auto& kv : by_tool) {
    std::map<std::string,int> votes;
    for (const auto& p : kv.second) {
      std::string reply; float sim = 0;
      if (am_query_public("build\\ckpt\\canticle.am", p, reply, sim) != 0) continue;
      const size_t a = reply.find("<|tool|>");
      std::string got = "none";
      if (a != std::string::npos) {
        size_t b = reply.find(' ', a);
        got = reply.substr(a + 8, (b == std::string::npos ? reply.size() : b) - (a + 8));
      }
      votes[got]++;
    }
    std::string best = "none"; int bestn = 0;
    for (const auto& v : votes) if (v.second > bestn) { best = v.first; bestn = v.second; }
    baseline[kv.first] = best;
    std::printf("  %-11s -> %-11s (%d/%zu agree)%s\n", kv.first.c_str(), best.c_str(),
                bestn, kv.second.size(),
                (best == kv.first) ? "" : "   <-- MISMATCH WITH ITS OWN CORPUS");
  }

  // ---- pairwise combinations ----
  //
  // Each pair is formed by taking a prompt from tool A and appending a prompt
  // fragment from tool B. The result should route to one of the two, never to a
  // third thing.
  std::printf("\n-- pairwise combinations that route to neither input --\n");
  int checked = 0, suspicious = 0;
  std::vector<std::string> findings;

  std::vector<std::string> keys;
  for (const auto& kv : by_tool) keys.push_back(kv.first);

  for (size_t i = 0; i < keys.size(); ++i) {
    for (size_t j = 0; j < keys.size(); ++j) {
      if (i == j) continue;
      const std::string& ta = keys[i];
      const std::string& tb = keys[j];
      for (size_t k = 0; k < std::min<size_t>(3, by_tool[ta].size()); ++k) {
        for (size_t m = 0; m < std::min<size_t>(2, by_tool[tb].size()); ++m) {
          const std::string combined = by_tool[ta][k] + " and " + by_tool[tb][m];
          std::string reply; float sim = 0;
          if (am_query_public("build\\ckpt\\canticle.am", combined, reply, sim) != 0) continue;
          const size_t ra = reply.find("<|tool|>");
          std::string got = "none";
          if (ra != std::string::npos) {
            size_t rb = reply.find(' ', ra);
            got = reply.substr(ra + 8, (rb == std::string::npos ? reply.size() : rb) - (ra + 8));
          }
          checked++;
          // A refusal is not a routing result, and the FIRST version of this test
          // got that wrong in a way that produced six false positives.
          //
          // The refusal message names the tools it DOES support:
          //     "I can: read a file, list a directory, write a file, ..."
          // and the extractor below scans for "<|tool|>", so a refusal yielded
          // "none" ... except that the multi-action message quotes the request
          // back, and the quoted text can contain a path that the downstream
          // matcher reads as a tool. The reliable test is the absence of a TOOL
          // CALL, not the presence of particular words.
          //
          // Stated plainly: classify on what the system DID, not on what it said.
          const bool produced_call = (ra != std::string::npos);
          const bool refused = !produced_call;
          if (!refused && got != baseline[ta] && got != baseline[tb]) {
            ++suspicious;
            findings.push_back("  " + ta + " + " + tb + "  ->  " + got +
                               "\n        a : " + by_tool[ta][k] +
                               "\n        b : " + by_tool[tb][m]);
          }
        }
      }
    }
  }

  for (const auto& f : findings) std::printf("%s\n", f.c_str());
  std::printf("\n  %d combinations checked, %d routed to a tool neither input uses\n",
              checked, suspicious);

  // ---------------------------------------------------------------------------
  //  ASSERT, not merely report.
  //
  //  As a report this tool is useful once and then decays: a later run with a
  //  non-zero count requires someone to notice. As an assertion it fails the build
  //  the moment a combination starts routing somewhere new.
  //
  //  The expectation of zero is justified by measurement rather than optimism. The
  //  hunt originally found six, all traced to one cause (the multi-action guard
  //  lived in the chat handler instead of the shared router, defect 66). With that
  //  fixed every combination routes to one of its two inputs or refuses -- so zero
  //  is the current, established behaviour, and any future non-zero count is a
  //  change worth stopping for.
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  //  A MINIMUM WORK ASSERTION, without which this suite passes by doing nothing.
  //
  //  Measured by an independent audit: run from a directory where the relative
  //  memory path does not resolve, every query fails, every combination is skipped
  //  by the "continue" on error, and the tool prints
  //      "0 combinations checked, 0 routed to a third tool"
  //      ALL PASS
  //  with exit 0. Worse, the baseline loop also continues on failure, so
  //  baseline[ta] is an empty string and "got != baseline[ta]" is TRUE for every
  //  got -- the harness becomes EASIER to pass as it breaks.
  //
  //  Zero suspicious findings out of zero attempts is not evidence. The threshold is
  //  set well below the measured 432 so ordinary corpus drift does not trip it,
  //  while a broken path or a broken memory file does.
  // ---------------------------------------------------------------------------
  if (checked < 200) {
    std::printf("\n  FAIL: only %d combinations were evaluated. A run this small cannot\n"
                "  distinguish correct behaviour from a harness that evaluated nothing.\n"
                "  Check that the memory path resolves and that the corpus is non-empty.\n",
                checked);
    return 1;
  }
  if (suspicious != 0) {
    std::printf("\n  FAIL: %d combination(s) route to a third tool. Each is listed\n",
                suspicious);
    std::printf("  above; decide whether the pair is genuinely ambiguous (then extend\n");
    std::printf("  the refusal rules) or a precedence error (then move the rule).\n");
    return 1;
  }
  std::printf("\n  ALL PASS  (%d combinations, none routed to a third tool)\n", checked);
  return 0;
}

// ===========================================================================
//  Multi-turn context hunt.
//
//  WHY THIS IS SEPARATE FROM am-inject
//  -----------------------------------
//  am-inject varies the SIGNALS inside one turn. This varies the CONTEXT across
//  turns, which is a different mechanism: Session::rewrite borrows the previous
//  turn's tool and payload to repair a context-dependent turn (the fix for gap B).
//
//  The property being checked is the one gap B promised:
//
//      a follow-up that names only a new payload keeps the previous turn's tool.
//
//  The natural way to break it is a rule change that makes the follow-up look
//  self-contained. That already happened once (defect 15): the bare-path default
//  claimed "and C:/work/b.txt" as a read, so a follow-up to a directory listing
//  became a file read and multi-turn accuracy fell from 11/11 to 9/11.
//
//  So the hunt enumerates every (first turn, follow-up form) pair and asserts the
//  tool is unchanged. Follow-up forms are taken from the corpus's own cue words
//  plus the payload-only forms, so the coverage is generated rather than listed.
// ===========================================================================
#include "canticle/am.h"
#include "canticle/data.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

using namespace canticle;

namespace {

std::string tool_of(const std::string& reply) {
  const size_t a = reply.find("<|tool|>");
  if (a == std::string::npos) return "refused";
  const size_t b = reply.find(' ', a);
  return reply.substr(a + 8, (b == std::string::npos ? reply.size() : b) - (a + 8));
}

} // namespace

int am_turns_run(int argc, char** argv) {
  const std::string am = "build\\ckpt\\canticle.am";
  int limit = 200;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--limit" && i + 1 < argc) limit = atoi(argv[++i]);

  const auto corpus = synth_tool_data(limit * 20, 31337);

  std::map<std::string, std::vector<std::string>> by_tool;
  for (const auto& s : corpus) {
    const std::string t = tool_of(s.reply);
    if (by_tool[t].size() < 3) by_tool[t].push_back(s.prompt);
  }

  // ---- follow-up forms ----
  //
  // Each supplies a NEW payload (or a cue for the same one) and names no tool.
  //
  // PAYLOAD-KIND COMPATIBILITY, and getting this wrong produced most of the
  // remaining false positives.
  //
  // The first version applied every follow-up to every tool, including:
  //
  //     t1 "can you solve 52 + 61"      (calc)
  //     t2 "and C:/work/zeta.txt"       a PATH
  //
  // and reported the result as a broken guarantee. It is not: "and <path>" after
  // an arithmetic request has no sensible reading -- a calculation's payload is a
  // number, and no rule could inherit a path into it. The router's answer
  // (list_dir for the new path) is the only sane one.
  //
  // So the follow-up carries a payload KIND and is only applied to tools whose own
  // payload is of that kind. Cues ("again") are kind-agnostic and applied to all.
  enum Kind { K_ANY, K_PATH, K_CMD };
  struct Follow { const char* name; const char* text; Kind kind; };
  static const Follow kFollow[] = {
    {"same-cue",       "again",                       K_ANY},
    {"same-cue-2",     "same",                        K_ANY},
    {"another-cue",    "another",                     K_ANY},
    {"chinese-cue",    "\u518d\u6765\u4e00\u6b21", K_ANY},
    {"path-and",       "and C:/work/zeta.txt",        K_PATH},
    {"path-and-2",     "and C:/log/omega.md",         K_PATH},
    {"path-bare",      "C:/tmp/iota.json",            K_PATH},
    {"path-and-3",     "and D:/data/kappa.log",       K_PATH},
    {"cmd-and",        "and Get-Location",            K_CMD},
    {"cmd-bare",       "Get-Date",                    K_CMD},
  };

  std::printf("=== multi-turn context hunt ===\n\n");

  // ---- first-turn baseline: what each tool's own prompt produces alone ----
  std::map<std::string, std::string> base;
  std::printf("-- first-turn baselines --\n");
  for (const auto& kv : by_tool) {
    std::string reply; float sim = 0;
    am_query_public(am, kv.second[0], reply, sim);
    base[kv.first] = tool_of(reply);
    std::printf("  %-11s -> %-11s | %s\n", kv.first.c_str(), base[kv.first].c_str(),
                kv.second[0].c_str());
  }

  // ---------------------------------------------------------------------------
  //  THE HARNESS MUST GO THROUGH Session, and the first version did not.
  //
  //  It called am_query_public twice, which retrieves each turn IN ISOLATION.
  //  Inheritance is exactly what that path does not do -- the mechanism lives in
  //  Session::rewrite, which the REPL calls before retrieval. So the harness
  //  reported 50+ "failures" that were artifacts of never invoking the feature it
  //  was meant to test.
  //
  //  This is the THIRD time a harness bug has produced false results (defects 67
  //  and 69 were the first two: sampling bias, and classifying on words instead of
  //  on produced calls). The pattern is consistent enough to state as a rule:
  //
  //      a test harness must drive the SAME code path as production, or it
  //      measures the harness.
  //
  //  So each pair now runs through a Session exactly as am-chat does: push the
  //  first turn (recording the tool and payload it produced), rewrite the second
  //  turn through that session, and route the REWRITTEN text.
  // ---------------------------------------------------------------------------
  std::printf("\n-- follow-ups that do not inherit the previous tool --\n");
  int checked = 0, failures = 0;
  std::vector<std::string> findings;

  for (const auto& kv : by_tool) {
    const std::string& tool = kv.first;
    if (base[tool] != tool) continue;   // first turn must be right for the pair to mean anything
    for (size_t k = 0; k < by_tool[tool].size(); ++k) {
      for (const Follow& f : kFollow) {
        // payload-kind gate: a path follow-up is meaningless after arithmetic,
        // and a command follow-up is meaningless after a file read.
        const bool path_tool = (tool == "read_file" || tool == "list_dir" ||
                                tool == "write_file" || tool == "stat" ||
                                tool == "mkdir");
        const bool cmd_tool  = (tool == "run_shell");
        if (f.kind == K_PATH && !path_tool) continue;
        if (f.kind == K_CMD  && !cmd_tool)  continue;

        const std::string t1 = by_tool[tool][k];
        const std::string t2 = f.text;

        // --- turn 1, through the real path ---
        std::string reply1; float s1 = 0;
        am_query_public(am, t1, reply1, s1);
        const std::string got1 = tool_of(reply1);
        if (got1 != tool) continue;

        // --- record the turn in a Session, as the REPL does ---
        Session sess;
        sess.push(t1, got1, extract_payload(t1));

        // --- turn 2: rewrite through the session, then route the result ---
        const std::string rewritten = sess.rewrite(t2);
        std::string reply2; float s2 = 0;
        if (am_query_public(am, rewritten, reply2, s2) != 0) continue;
        const std::string got = tool_of(reply2);

        checked++;
        if (got == tool) continue;
        if (got == "refused") continue;   // a legitimate refusal is not a failure
        ++failures;
        findings.push_back("  " + tool + " + follow:" + f.name + "  ->  " + got +
                           "\n        t1  : " + t1 +
                           "\n        t2  : " + t2 +
                           "\n        rw  : " + rewritten);
      }
    }
  }

  for (const auto& x : findings) std::printf("%s\n", x.c_str());
  std::printf("\n  %d follow-ups checked, %d switched tool\n", checked, failures);
  if (failures) {
    std::printf("\n  FAIL: each case above breaks the gap-B guarantee that a\n");
    std::printf("  payload-only follow-up keeps its predecessor's tool.\n");
    return 1;
  }

  // ---------------------------------------------------------------------------
  //  THE REBUILD LEAD IS EXERCISED, which nothing tested before.
  //
  //  Mutation testing found this hole: setting read_file's lead to nullptr changed
  //  no suite's verdict. The loop above checks whether the TOOL survives a
  //  follow-up, and the tool survives either way -- with a lead the rewritten text
  //  is "read the file <payload>", without one the previous prompt is reused
  //  verbatim. Both keep the tool, so that assertion could not see the difference.
  //
  //  What only a lead can do is RE-AIM the request at a new payload, so that is the
  //  property asserted: a single-argument tool given a new payload must produce a
  //  rewritten text containing it. Without the lead the rewrite is the old prompt
  //  and the new payload is absent.
  // ---------------------------------------------------------------------------
  std::printf("-- the rebuild lead actually re-aims the request --\n");
  {
    struct LeadCase { const char* tool; const char* first; const char* follow;
                      const char* expect_payload; };
    static const LeadCase kCases[] = {
      {"read_file", "read the file C:/tmp/a.txt", "and C:/tmp/zeta.txt", "zeta.txt"},
      {"list_dir",  "list the directory C:/tmp",  "and C:/work",         "C:/work"},
      {"run_shell", "run the command hostname",   "and whoami",          "whoami"},
      {"calc",      "what is 40 - 14",            "and 60 - 14",         "60"},
    };
    for (const LeadCase& c : kCases) {
      std::string r1; float s1 = 0;
      if (am_query_public(am, c.first, r1, s1) != 0) continue;
      Session sess;
      sess.push(c.first, c.tool, extract_payload(c.first));
      const std::string rw = sess.rewrite(c.follow);
      checked++;
      if (rw.find(c.expect_payload) == std::string::npos) {
        std::printf("  FAIL %-11s follow-up \"%s\" did not carry \"%s\" into the rewrite\n"
                    "        got: %.70s\n", c.tool, c.follow, c.expect_payload, rw.c_str());
        failures++;
      }
    }
  }

  // ---------------------------------------------------------------------------
  //  THE REWRITE MUST ACTUALLY HAPPEN, not merely produce the right tool.
  //
  //  Mutation testing found this: making Session::rewrite return its input unchanged
  //  changed no verdict, because a bare payload like "C:/work/zeta.txt" is often
  //  close enough to the right template that retrieval picks the same tool anyway.
  //  The checks above measured the ANSWER, not the MECHANISM, so a rewrite that did
  //  nothing passed them all.
  //
  //  What the mechanism guarantees is textual: for a context-dependent turn the
  //  rewritten text DIFFERS from the input and names the inherited tool.
  // ---------------------------------------------------------------------------
  std::printf("-- the rewrite changes the text, not just the outcome --\n");
  {
    struct RwCase { const char* tool; const char* first; const char* follow; };
    static const RwCase kCases[] = {
      {"read_file", "read the file C:/tmp/a.txt", "again"},
      {"list_dir",  "list the directory C:/tmp",  "and C:/work"},
      {"calc",      "what is 40 - 14",            "again"},
      {"run_shell", "run the command hostname",   "and whoami"},
    };
    for (const RwCase& c : kCases) {
      Session sess;
      sess.push(c.first, c.tool, extract_payload(c.first));
      const std::string rw = sess.rewrite(c.follow);
      checked++;
      if (rw == c.follow) {
        std::printf("  FAIL %-11s \"%s\" returned UNCHANGED; no inheritance happened\n",
                    c.tool, c.follow);
        failures++;
        continue;
      }
      checked++;
      std::string low = rw;
      for (char& ch : low) ch = (char)std::tolower((unsigned char)ch);
      const std::string tl = c.tool;
      const bool names_tool =
          (tl == "read_file" && low.find("read")    != std::string::npos) ||
          (tl == "list_dir"  && low.find("list")    != std::string::npos) ||
          (tl == "calc"      && low.find("what is") != std::string::npos) ||
          (tl == "run_shell" && low.find("run ")    != std::string::npos);
      if (!names_tool) {
        std::printf("  FAIL %-11s rewrite does not name the inherited tool: %.60s\n",
                    c.tool, rw.c_str());
        failures++;
      }
    }
  }

  // Minimum work. On a fresh clone with no memory every baseline is "refused", the
  // follow-up loop finds nothing to check, and this suite reported ALL PASS after
  // evaluating 8 checks -- all of them the hand-written ones, none of the corpus.
  // A run that measured nothing must not report a verdict (defect 105).
  if (checked < 100) {
    std::printf("\n  FAIL: only %d checks ran. The corpus loop evaluated nothing,\n"
                "  which usually means the memory is missing.\n"
                "  Build one first: canticle am-build --data 6000\n", checked);
    return 1;
  }
  if (failures) return 1;
  std::printf("\n  ALL PASS  (%d checks: follow-ups inherit, leads re-aim, rewrites happen)\n", checked);
  return 0;
}

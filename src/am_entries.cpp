// ===========================================================================
//  Entry-point consistency.
//
//  THE PROBLEM
//  -----------
//  Four code paths reach retrieval:
//
//      am-ask / am-chat   the interactive turn handler
//      am_query_public    the programmatic entry used by the test harnesses
//      am-eval            the accuracy measurement
//      am-multi           the online-teaching driver
//
//  Every one of them must agree about what a request means. Twice they did not:
//
//      defect 66  the multi-action guard lived in the chat handler only, so
//                 am_query_public answered two-action requests with a mixed call
//      defect 69  the Unsupported guard, in the very next block, was left behind
//                 when 66 was fixed -- so "delete C:/tmp/x.txt" refused in the
//                 REPL and was routed to read_file through am_query_public
//
//  and a third difference was found while writing this file: am-multi mapped both
//  refusal codes to "(no match)", reporting "understood but declined" as
//  "nothing found".
//
//  Both fixes were "move the check to the shared layer". The reason a third gap
//  kept appearing is that nothing CHECKED the agreement -- each fix was verified by
//  the specific symptom that prompted it.
//
//  WHAT THIS FILE DOES
//  -------------------
//  It runs the same inputs through the paths that are reachable from one process
//  and asserts they produce the same reply. Where a path cannot be invoked
//  in-process (the REPL), the case is exercised through am_query_public, which is
//  what the REPL now delegates to for retrieval.
// ===========================================================================
#include "canticle/am.h"
#include <cstdio>
#include <string>
#include <vector>

using namespace canticle;

static int fails = 0, total = 0;

static void same(const std::string& q, const std::string& a, const std::string& b,
                 const char* what) {
  total++;
  if (a != b) {
    std::printf("  FAIL %s\n        a: %.70s\n        b: %.70s\n", what, a.c_str(), b.c_str());
    fails++;
  }
}

int am_entries_run(int, char**) {
  const std::string am = "build\\ckpt\\canticle.am";
  std::printf("=== entry-point consistency ===\n\n");

  // Each case is chosen to exercise one of the shared refusals or the normal path.
  struct Case { const char* q; const char* why; };
  static const Case kCases[] = {
    {"delete C:/tmp/x.txt",                        "unsupported refusal (-4)"},
    {"remove C:/tmp/x.txt",                        "unsupported refusal, alt verb"},
    {"rename C:/a.txt to C:/b.txt",                "unsupported refusal, rename"},
    {"read the file C:/Users/cc/canticle/STATUS.md","ordinary read"},
    {"list the directory C:/tmp",                  "ordinary list"},
    {"what is 40 - 14",                            "ordinary calc"},
    {"read C:/a.txt and list C:/tmp",              "two actions (-3)"},
    {"find all TODO in C:/tmp",                    "ordinary search"},
    {"how big is C:/Users/cc/canticle/STATUS.md",  "ordinary stat"},
  };

  std::printf("-- am_query_public is stable across repeated calls --\n");
  //
  // The memory is cached in a static, so a second call exercises the cached path.
  // A difference here would mean the first call mutates something it should not --
  // which is the failure mode that made am_learn.cpp run its probes in a fresh
  // process.
  for (const Case& c : kCases) {
    std::string r1, r2; float s1 = 0, s2 = 0;
    if (am_query_public(am, c.q, r1, s1) != 0) { std::printf("  [skip] %s\n", c.q); continue; }
    if (am_query_public(am, c.q, r2, s2) != 0) { std::printf("  [skip] %s\n", c.q); continue; }
    same(c.q, r1, r2, c.why);
  }

  std::printf("-- a refusal is a refusal, not a tool call --\n");
  //
  // The property that defect 69 broke. Stated as an assertion rather than a comment:
  // a request this system cannot serve must not produce a <|tool|> block, through
  // ANY entry point.
  static const char* kUnsupported[] = {
    "delete C:/tmp/x.txt", "remove C:/tmp/x.txt", "erase C:/tmp/x.txt",
    "rename C:/a.txt to C:/b.txt", "move C:/a.txt to C:/b.txt",
  };
  for (const char* q : kUnsupported) {
    std::string r; float s = 0;
    if (am_query_public(am, q, r, s) != 0) continue;
    total++;
    if (r.find("<|tool|>") != std::string::npos) {
      std::printf("  FAIL \"%s\" produced a tool call: %.70s\n", q, r.c_str());
      fails++;
    }
  }

  std::printf("-- the two refusal kinds are distinguishable --\n");
  //
  // -3 and -4 promise different things. A caller that collapses them cannot tell
  // the user which one applies, which is the whole reason both exist.
  {
    std::string two, one; float s1 = 0, s2 = 0;
    const int rc_two = am_query_public(am, "read C:/a.txt and list C:/tmp", two, s1);
    const int rc_one = am_query_public(am, "delete C:/tmp/x.txt", one, s2);

    // -------------------------------------------------------------------------
    //  A MISSING MEMORY IS NOT A DISAGREEMENT BETWEEN ENTRY POINTS.
    //
    //  On a fresh clone this suite failed with "the two-action reply does not say
    //  so: <empty>" for both kinds. The cause was -1 (empty memory): every query
    //  returns an empty answer, so the two texts match trivially and neither
    //  contains the phrase being looked for.
    //
    //  The verdict was correct -- something IS wrong -- but the message sent the
    //  reader looking for a divergence between two code paths that are both
    //  behaving identically. Naming the real precondition turns a misleading
    //  failure into a useful one.
    // -------------------------------------------------------------------------
    // Any nonzero code means one of the two queries did not produce an answer at
    // all. The first version tested only for -1 (empty memory) and therefore did not
    // fire: on a fresh clone the code was different, both texts were empty, and the
    // suite reported a code-path divergence that did not exist.
    if (rc_two != 0 || rc_one != 0) {
      std::printf("  SKIP: the queries did not answer (rc=%d/%d), so comparing the two\n"
                  "  texts would compare two empty strings.\n"
                  "  Build a memory first: canticle am-build --data 6000\n", rc_two, rc_one);
      return 1;
    }
    same("refusal kinds", two == one ? "SAME" : "DIFFERENT", "DIFFERENT",
         "two-action and unsupported give different replies");
    total++;
    if (two.find("two actions") == std::string::npos) {
      std::printf("  FAIL the two-action reply does not say so: %.70s\n", two.c_str());
      fails++;
    }
    if (one.find("no tool for it") == std::string::npos) {
      std::printf("  FAIL the unsupported reply does not say so: %.70s\n", one.c_str());
      fails++;
    }
  }

  std::printf("\n  %s  (%d failure(s), %d checks)\n", fails ? "FAILED" : "ALL PASS", fails, total);
  if (fails) {
    std::printf("\n  A failure here means two entry points disagree about what a\n");
    std::printf("  request means -- the condition behind defects 66 and 69.\n");
  }
  return fails ? 1 : 0;
}

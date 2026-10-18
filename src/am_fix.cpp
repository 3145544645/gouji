// ===========================================================================
//  Learning from failure.
//
//  WHAT MAKES THIS DIFFERENT FROM am-learntest
//  -------------------------------------------
//  am-learntest covers teaching by explicit instruction: a human says "this is
//  the answer". This file covers the other path -- the system noticing its own
//  mistake and absorbing the user's fix.
//
//  The mechanism: a turn that FAILS is remembered; the next successful turn, if
//  it shares content with the failed request, is treated as the correction and
//  the working action is attached to the failed request.
//
//  TWO PROPERITES THIS FILE PINS DOWN, both of which were broken at some point:
//
//    1. RECALL -- after one correction the failed request works, in a fresh
//       process. Without this the pairing machinery is decoration.
//
//    2. NO FALSE PAIRING -- a failed turn followed by an UNRELATED request must
//       not learn anything. The first implementation paired on time adjacency
//       alone, so "play some jazz" (failed) followed by "what is 7 * 8" taught
//       the memory to answer "play some jazz" with calc 7*8. It persisted.
// ===========================================================================
#include "canticle/am.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

namespace canticle {

struct FixCase {
  const char* turn1;        // expected to fail
  const char* turn2;        // the user's follow-up
  bool        expect_pair;  // should turn2 be taken as a correction
  const char* probe;        // asked afterwards
  const char* want;         // substring the probe should produce ("" = unchanged)
  const char* note;
};

// NOTE ON THESE INPUTS, after the similarity floor was introduced.
//
// The cases used to open with "play some jazz", which the system routed to
// read_file with a fabricated path. With the floor in place that request is now
// REFUSED ("I do not have a tool for that") -- better behaviour, but it means the
// refusal path never reaches the tool layer, so the execution failure this test
// needs never happens and no correction can be paired.
//
// The turns below are therefore chosen to have HIGH similarity (so they are acted
// on) while still FAILING at execution. That is also the more realistic scenario:
// a request understood well enough to attempt, and got wrong.
static const FixCase kFix[] = {
  // A correction that RESTATES the noun ("hostname" appears in both turns) is
  // paired. This is the realistic shape: a user fixing a misroute keeps the thing
  // they were asking for and changes the verb.
  // The realistic shape of a correction: the user keeps the SUBJECT and changes the
  // operation. Here they asked to read something that is actually a directory --
  // turn 1 fails at execution -- then asked to list it instead, which succeeds and
  // shares the content word "windows".
  //
  // Note that the corrected action need not be one this system invented: it only
  // has to succeed, because the pairing rule copies the action that WORKED.
  {"read the file C:/Windows", "list the directory C:/Windows", true,
   "read the file C:/Windows", "",
   "correction sharing a content word is paired"},

  // LIMITATION, measured every run rather than described in a comment.
  //
  // Even with the pairing persisted (the delta file holds
  // "read the file C:/Windows" and the memory reports one more episode), a fresh
  // query for that exact string still returns the old answer. The new row's
  // similarity to the query loses to rows like "read the file C:/log/log.txt".
  //
  // The cause is structural, not a defect in the pairing: inside the read_file
  // family every stored prompt differs only in its path, so a path the encoder has
  // never seen scores below paths shaped like the family average. One new row
  // cannot outvote three thousand near-identical ones.
  //
  // Pinned so the boundary is visible. Relaxing it would need retrieval that can
  // prefer a recently taught row, which is a different mechanism from similarity.
  // Previously pinned as a limitation: the pairing persisted but retrieval kept
  // returning the old answer, and the cause had been assumed to be structural --
  // one row cannot outvote three thousand near-identical ones.
  //
  // That diagnosis was WRONG. The row was filed in the bucket of its REPLY's tool
  // (list_dir) while the search ran in the bucket of the PROMPT's inferred intent
  // (read_file), so it was never a candidate. It scored 1.000 against its own query
  // once the bucket was corrected -- it had never been losing anything.
  //
  // Now asserted as a pass, because the behaviour is real capability rather than a
  // boundary: the taught answer is retrieved and returned.
  {"read the file C:/Windows", "list the directory C:/Windows", true,
   "read the file C:/Windows", "list_dir",
   "corrected answer is retrieved, not merely stored"},

  // LIMITATION, pinned deliberately rather than hidden.
  //
  // A correction that changes BOTH the verb and the noun shares no content word,
  // so the relevance gate rejects it and nothing is learned. This is the price of
  // the gate that prevents the false pairing documented in defect 24, where an
  // unrelated follow-up was absorbed as a correction and the wrong answer was
  // persisted permanently.
  //
  // Between the two failures -- occasionally missing a real correction, versus
  // permanently learning a wrong one -- missing it is the recoverable outcome:
  // the user can correct again, whereas a poisoned memory needs the delta file
  // to be deleted. The gate therefore stays, and the limitation is recorded here
  // so that a future attempt to relax it starts from the evidence.
  {"read the file C:/Windows", "run the command hostname", false,
   "read the file C:/Windows", "",
   "LIMITATION: correction with no shared content word is not paired"},
  {"read the file C:/Windows", "what is 7 * 8", false,
   "read the file C:/Windows", "",
   "unrelated follow-up is NOT learned"},
  // NEGATIVE: repeating the identical failing request is not a correction. If the
  // same string succeeds the difference was environmental (the file appeared), and
  // learning it would attach an unrelated action to this request.
  {"read the file C:/Windows", "read the file C:/Windows", false,
   "read the file C:/Windows", "",
   "identical repeat is environmental, not a correction"},
};

} // namespace canticle

using namespace canticle;

int am_fix_run(int argc, char** argv) {
  const int N = (int)(sizeof(kFix) / sizeof(kFix[0]));
  std::printf("=== learning from failure ===\n");
  std::printf("turn 1 fails; turn 2 either is the fix or is a topic change\n\n");

  int ok = 0;
  for (int i = 0; i < N; ++i) {
    const std::string am = "build\\ckpt\\canticle.am";
    std::remove((am + ".queue").c_str());
    std::remove((am + ".delta").c_str());

    // ---- run the two turns ----
    std::string script = std::string(kFix[i].turn1) + "|" + kFix[i].turn2;
    std::string c1 = "build\\canticle.exe am-chat --learn --script \"" + script
                   + "\" > build\\ckpt\\fix.txt 2>&1";
    std::system(c1.c_str());

    std::ifstream f("build\\ckpt\\fix.txt", std::ios::binary);
    std::stringstream ss; ss << f.rdbuf();
    const std::string log = ss.str();
    const bool paired = log.find("correction pairing") != std::string::npos;

    // ---- apply whatever was admitted ----
    std::string c2 = "build\\canticle.exe am-harvest --apply > nul 2>&1";
    std::system(c2.c_str());

    // ---- probe in a fresh process ----
    std::string c3 = "build\\canticle.exe am-chat --prompt \"" + std::string(kFix[i].probe)
                   + "\" > build\\ckpt\\fixp.txt 2>&1";
    std::system(c3.c_str());
    std::ifstream g("build\\ckpt\\fixp.txt", std::ios::binary);
    std::stringstream ss2; ss2 << g.rdbuf();
    const std::string out = ss2.str();
    std::string got;
    const size_t p = out.find("canticle>");
    if (p != std::string::npos) {
      size_t e = out.find('\n', p);
      got = out.substr(p + 9, (e == std::string::npos ? out.size() : e) - p - 9);
    }

    // ---- judge ----
    //
    // A case whose note begins with "LIMITATION" asserts that the gate REJECTS a
    // particular correction, and that is the expected outcome rather than a
    // failure. It is pinned in the same table so that the boundary is measured
    // every run instead of being described in a comment and silently changing.
    // Four kinds of case, distinguished by the note and the expectation:
    //
    //   expect_pair && want non-empty  the correction is learned AND retrieval
    //                                  reflects it
    //   expect_pair && want empty      the correction is learned, but retrieval
    //                                  is not asserted -- used where persistence is
    //                                  real yet a single row cannot win retrieval
    //   !expect_pair && LIMITATION     the gate is expected to REJECT: the boundary
    //                                  is measured rather than merely documented
    //   !expect_pair                   nothing may be learned from an unrelated
    //                                  follow-up
    bool hit;
    const bool is_limitation  = (std::strncmp(kFix[i].note, "LIMITATION", 10) == 0);
    const bool is_rlimit      = (std::strncmp(kFix[i].note, "RLIMIT", 6) == 0);
    if (is_limitation) {
      hit = !paired;                      // the gate rejects, and that is correct
    } else if (is_rlimit) {
      // Pairing works AND retrieval does not reflect it. Both halves are asserted,
      // so if retrieval ever starts honouring a single new row this case fails and
      // the limitation is revisited rather than silently becoming stale.
      hit = paired && (got.find(kFix[i].want) == std::string::npos);
    } else if (kFix[i].expect_pair) {
      hit = paired && (kFix[i].want[0] == '\0' ||
                       got.find(kFix[i].want) != std::string::npos);
    } else {
      hit = !paired;
    }
    if (hit) ok++;

    std::printf("  %s %s\n", hit ? "[ok]  " : "[MISS]", kFix[i].note);
    std::printf("        turn1 : %s\n", kFix[i].turn1);
    std::printf("        turn2 : %s\n", kFix[i].turn2);
    std::printf("        paired: %s   probe -> %s\n",
                paired ? "yes" : "no", got.c_str());
  }

  const std::string am = "build\\ckpt\\canticle.am";
  std::remove((am + ".queue").c_str());
  std::remove((am + ".delta").c_str());
  std::printf("\n  OVERALL  %d/%d  %.1f%%\n", ok, N, 100.0 * ok / N);
  return ok == N ? 0 : 1;
}

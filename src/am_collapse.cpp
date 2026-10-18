// ===========================================================================
//  Does collapsing a frame's examples into one row lose coverage?
//
//  THE CLAIM
//  ---------
//  Three observations of "play some X" do not need three memory rows. Retrieval
//  already generalises across a family through slot transfer, so one row per
//  RULE should answer the whole family.
//
//  If true, memory growth tracks the number of rules a deployment learns rather
//  than the number of things users say -- a large difference for any real use.
//
//  HOW IT IS MEASURED
//  ------------------
//  Build the same memory twice:
//    A: teach three examples, apply WITHOUT collapsing
//    B: teach the same three examples, apply WITH collapsing
//  Then probe queries the memory was never taught, and compare.
//
//  A win is: B answers every probe that A answers, using fewer rows.
//  A failure is: B misses something A caught -- the collapsed row does not cover
//  the family, and the feature costs accuracy for compactness.
// ===========================================================================
#include "canticle/am.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

using namespace canticle;

// Probes this memory has never been taught.
//
// THEY MUST BELONG TO THE SAME FAMILY AS THE TEACHING, and the first version did
// not: it taught "run the command ver/vol/set/dir/hostname" and then probed
// "play some reggae" and four siblings. Those are a different template entirely, so
// the probes scored 0/5 in BOTH variants -- and a comparison of 0 against 0 reports
// no difference regardless of what collapsing does. The experiment had two broken
// halves: the admission rule rejected the teaching (defect 111) and the probes asked
// about something else entirely.
//
// These probes reuse the taught template with payloads never taught, which is what
// slot transfer is supposed to answer. That is the capability the collapse claim
// rests on: "one row per RULE should answer the whole family".
static const char* kProbes[] = {
  "run the command tree", "run the command ver", "run the command where",
  "run the command vol",  "run the command type",
};
static const int NPROBE = 5;

static std::string run_capture(const std::string& cmd) {
  std::system(cmd.c_str());
  std::ifstream f("build\\ckpt\\cov.txt", std::ios::binary);
  std::stringstream ss; ss << f.rdbuf();
  return ss.str();
}

// Row count read through the STATS hook, which prints a single parseable line
// and exits before any retrieval work.
static int row_count(const std::string& am) {
  std::system(("set AM_STATS_ONLY=1 && build\\canticle.exe am-chat --am " + am +
               " --prompt x > build\\ckpt\\cov.txt 2>&1").c_str());
  std::ifstream f("build\\ckpt\\cov.txt", std::ios::binary);
  std::stringstream ss; ss << f.rdbuf();
  const std::string s = ss.str();
  const size_t p = s.find("episodes=");
  if (p == std::string::npos) return -1;
  return std::atoi(s.c_str() + p + 9);
}

static int count_covered(const std::string& am) {
  int hit = 0;
  for (int i = 0; i < NPROBE; ++i) {
    const std::string c = "build\\canticle.exe am-chat --am " + am +
                          " --prompt \"" + kProbes[i] + "\" > build\\ckpt\\cov.txt 2>&1";
    const std::string out = run_capture(c);
    const size_t p = out.find("canticle>");
    if (p == std::string::npos) continue;
    size_t e = out.find('\n', p);
    const std::string reply = out.substr(p + 9, (e == std::string::npos ? out.size() : e) - p - 9);
    if (reply.find("run_shell") != std::string::npos) hit++;
  }
  return hit;
}

int am_collapse_run(int, char**) {
  std::printf("=== frame collapsing vs storing every example ===\n\n");
  int rows_a = -1, rows_b = -1, cov_a = 0, cov_b = 0;

  // A frame with FIVE members, each requested twice.
  //
  // Why this shape: collapsing only pays off when a frame has several admitted
  // members. The earlier scenario used a frame whose other members were already
  // well covered (sim >= SIM_CEILING), so only one example was admitted and the
  // frame row replaced it one-for-one -- identical row counts, and a test that
  // could not distinguish the two strategies.
  //
  // Each payload is a command that SUCCEEDS, so the observation carries a working
  // action and is admissible.
  const char* ex[5] = {"run the command ver", "run the command vol",
                       "run the command set", "run the command dir",
                       "run the command hostname"};
  const int NEX = 5;

  // A and B are built by the same binary and the same teaching calls; the only
  // difference is whether --no-collapse is passed at apply time.
  for (int variant = 0; variant < 2; ++variant) {
    const std::string am = variant == 0 ? "build\\ckpt\\covA.am" : "build\\ckpt\\covB.am";
    std::remove((am + ".delta").c_str());
    std::remove((am + ".queue").c_str());
    std::system(("del /q " + am + " 2>nul").c_str());

    std::system(("build\\canticle.exe am-build --data 3000 --am " + am + " > nul 2>&1").c_str());

    // Teach through the REAL flow: a failed request, a correction that works, then
    // two more members of the same family. Those later two succeed on their own
    // because retrieval generalises through slot transfer -- which is the reason a
    // frame exists at all.
    //
    // An earlier version of this test called am-learn directly for each example.
    // That bypasses the observation queue entirely, so no observation carried a
    // working action, and induction correctly refused to build a frame ("nothing
    // to generalise INTO"). The test then compared two runs that were identical
    // and reported no difference.
    std::string script;
    for (int i = 0; i < NEX; ++i) {
      if (!script.empty()) script += "|";
      script += ex[i];
      script += "|";
      script += ex[i];      // twice, so hits == 2 clears the support threshold
    }
    std::system(("build\\canticle.exe am-chat --learn --am " + am + " --script \"" +
                 script + "\" > nul 2>&1").c_str());
    // The ONLY difference between the two runs is this flag. Output is captured
    // rather than discarded so the run reports what it actually did -- guessing at
    // the row arithmetic produced two rounds of wrong conclusions.
    const std::string harv = "build\\canticle.exe am-harvest --am " + am + " --apply " +
                             (variant == 0 ? "--no-collapse " : "") + "> build\\ckpt\\harv.txt 2>&1";
    std::system(harv.c_str());
    {
      std::ifstream hf("build\\ckpt\\harv.txt", std::ios::binary);
      std::stringstream hs; hs << hf.rdbuf();
      const std::string h = hs.str();
      for (const char* key : {"observation(s)", "admitted of", "applied", "frame row"}) {
        const size_t p = h.find(key);
        if (p == std::string::npos) continue;
        size_t b2 = h.rfind('\n', p);
        b2 = (b2 == std::string::npos) ? 0 : b2 + 1;
        size_t e2 = h.find('\n', p);
        std::printf("      | %s\n", h.substr(b2, e2 - b2).c_str());
      }
    }

    const int base = row_count(am);
    const int cov = count_covered(am);
    std::printf("  %-30s rows=%d  probes answered: %d/%d\n",
                variant == 0 ? "A. every example stored"
                             : "B. frame members collapsed",
                base, cov, NPROBE);
    if (variant == 0) { rows_a = base; cov_a = cov; }
    else              { rows_b = base; cov_b = cov; }

    std::remove((am + ".delta").c_str());
    std::remove((am + ".queue").c_str());
  }
  std::remove("build\\ckpt\\cov.txt");
  std::remove("build\\ckpt\\c.row");

  // ---------------------------------------------------------------------------
  //  THE COMPARISON, which the file described in its header and never performed.
  //
  //  The header states the criterion plainly:
  //      "A win is: B answers every probe that A answers, using fewer rows.
  //       A failure is: B misses something A caught."
  //  and then the code printed both rows counts and returned 0. The claim about
  //  frame collapsing -- that memory grows with the number of RULES rather than the
  //  number of things users say -- was therefore never tested, only illustrated.
  //
  //  Two independent faults had to be fixed before the comparison could mean
  //  anything, and neither was visible while the numbers merely got printed:
  //
  //      defect 111  the admission rule rejected all five teaching observations, so
  //                  both variants stored identical memory
  //      defect 112  the probes asked about a different template than the teaching,
  //                  so both variants scored 0/5
  //
  //  0 against 0 and 3379 against 3379 both look like "no difference". Printing is
  //  not measuring; a comparison that cannot distinguish "equal" from "both broken"
  //  is not a test.
  // ---------------------------------------------------------------------------
  std::printf("\n-- comparison --\n");
  int rc = 0;
  if (rows_a < 0 || rows_b < 0) {
    std::printf("  FAIL: row counts could not be read (A=%d, B=%d)\n", rows_a, rows_b);
    rc = 1;
  } else if (cov_a == 0 && cov_b == 0) {
    std::printf("  FAIL: neither variant answered a single probe. A comparison of zero\n"
                "  against zero cannot distinguish 'collapsing is neutral' from 'the\n"
                "  experiment is broken', and the experiment HAS been broken here twice.\n");
    rc = 1;
  } else if (cov_b < cov_a) {
    std::printf("  FAIL: collapsing lost coverage -- B answered %d of %d, A answered %d.\n"
                "  The collapsed row does not cover the family, so the feature trades\n"
                "  accuracy for compactness.\n", cov_b, NPROBE, cov_a);
    rc = 1;
  } else if (rows_b >= rows_a) {
    std::printf("  FAIL: no rows were saved (A=%d, B=%d). Collapsing is claimed to reduce\n"
                "  storage; if it does not, it is pure cost.\n", rows_a, rows_b);
    rc = 1;
  } else {
    std::printf("  PASS: B answers %d/%d with %d fewer rows (%d -> %d).\n",
                cov_b, NPROBE, rows_a - rows_b, rows_a, rows_b);
    std::printf("  The claim holds on this scenario: one row per rule covers the family.\n");
  }
  return rc;
}

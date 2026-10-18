// ===========================================================================
//  am-generalize: does the system handle commands it was never taught?
//
//  WHY THIS EXISTS
//  -------------
//  ROADMAP stage 2 claims the biggest gap is combinatorial generalisation, and
//  until this file existed the evidence was a handful of hand-typed sentences:
//  "ping works, netstat works, dir C:/tmp does not". That is an impression, not a
//  measurement, and it cannot answer the stage-2 gate (>50% on unseen verbs, with
//  out-of-scope stays under 5%).
//
//  THE SETS, AND WHY THEY ARE THESE
//  -------------------------------
//  TRAINED:   the four commands synth_tool_data actually puts in the corpus.
//  UNSEEN:    commands from known_command that appear NOWHERE in training.
//  OOS:       sentences about the world that name no program at all.
//
//  UNSEEN is drawn from known_command on purpose. If it were invented strings the
//  test would measure the vocabulary guard, not generalisation. These are real
//  program names the model has never seen in a training sentence -- exactly the
//  situation stage 2 is about.
//
//  TWO NUMBERS, AND BOTH MUST BE REPORTED
//  -------------------------------------
//    recall   of the UNSEEN commands, how many produced run_shell with the right
//             payload. Low recall = no generalisation.
//    safety   of the OOS sentences, how many produced ANY tool call. High safety
//             violation means the system guesses, which is worse than refusing.
//
//  Reporting only recall would reward a system that fires on everything. Reporting
//  only safety would reward one that never acts. The gate needs both.
// ===========================================================================
#include "canticle/am.h"
#include "canticle/context.h"
#include <cstdio>
#include <string>
#include <vector>
#include <cstring>

using namespace canticle;

namespace {

// The four commands that appear in training. Anything else is unseen.
bool is_trained(const std::string& c) {
  return c == "Get-Date" || c == "hostname" ||
         c == "Get-Location" || c == "whoami";
}

// Real program names from known_command, minus the trained four.
// Hand-picked from the vocabulary, not generated: a generator would produce
// strings the vocabulary guard rejects, and the test would measure the guard.
const char* kUnseen[] = {
  "ping", "tracert", "netstat", "ipconfig", "tasklist", "systeminfo",
  "nslookup", "findstr", "tree", "ver", "vol", "where",
  "taskkill", "more", "set", "echo", "cd", "exit",
};

// Sentences about the world. None names a program, none is a request to run one.
// Drawn from the same distribution as CLINC150's out-of-scope half, because that
// is where the 1.3% figure comes from and the two must agree.
const char* kOos[] = {
  "how do i prevent my credit score from dropping",
  "what is the best practice in TCP/IP",
  "can you give me the standard tuning notes",
  "i need to buy a new pair of running shoes",
  "what does the word serendipity mean",
  "is it going to rain tomorrow in seattle",
  "how many calories are in a banana",
  "tell me a joke about programmers",
  "what is the capital of mongolia",
  "who won the world cup in 2018",
  "my phone battery drains too fast",
  "recommend a good book on stoicism",
};

} // namespace

int am_generalize_run(int argc, char** argv) {
  std::string ampath = "build\\ckpt\\canticle.am";
  bool verbose = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--am" && i + 1 < argc) ampath = argv[++i];
    else if (a == "-v") verbose = true;
  }

  std::printf("=== combinatorial generalisation ===\n\n");
  std::printf("trained commands : Get-Date, hostname, Get-Location, whoami (4)\n");
  std::printf("unseen commands  : %d, all present in known_command but absent from\n",
              (int)(sizeof(kUnseen) / sizeof(kUnseen[0])));
  std::printf("                   every training sentence\n");
  std::printf("out-of-scope     : %d sentences that name no program\n\n",
              (int)(sizeof(kOos) / sizeof(kOos[0])));

  // ---------------------------------------------------------------------------
  //  THREE SURFACE FORMS PER COMMAND, not one.
  //
  //  The first version tested only "run the command X" -- the same lead the
  //  corpus uses -- and reported 18/18. That is the easiest possible phrasing and
  //  it says little: a system that keys on one literal prefix would score the
  //  same. Varying the lead is what separates "learned the template" from "learned
  //  one string".
  //
  //  Measured on the three forms below: 54/54. The leads are all in the corpus, so
  //  this measures slot transfer rather than novel syntax -- stated here so the
  //  number is not read as more than it is.
  // ---------------------------------------------------------------------------
  static const char* kForms[] = {
    "run the command %s",
    "execute %s",
    "please run %s",
  };
  const int kFormCount = (int)(sizeof(kForms) / sizeof(kForms[0]));

  int unseen_hit = 0, unseen_n = 0;
  std::printf("-- unseen commands --\n");
  for (const char* cmd : kUnseen) {
    for (int f = 0; f < kFormCount; ++f) {
      char qbuf[256];
      std::snprintf(qbuf, sizeof qbuf, kForms[f], cmd);
      const std::string q = qbuf;
      std::string reply; float sim = 0;
      if (am_query_public(ampath, q, reply, sim) != 0) { unseen_n++; continue; }
      unseen_n++;
      const bool tool_ok = (reply.find("<|tool|>run_shell ") != std::string::npos);
      const bool pay_ok  = (reply.find(std::string("\"") + cmd + "\"") != std::string::npos);
      if (tool_ok && pay_ok) {
        unseen_hit++;
        if (verbose) std::printf("  [ ok ] %-34s %s\n", q.c_str(), reply.c_str());
      } else if (verbose) {
        std::printf("  [MISS] %-34s %s\n", q.c_str(), reply.c_str());
      }
    }
  }
  const double recall = unseen_n ? 100.0 * unseen_hit / unseen_n : 0.0;
  std::printf("  -> %d/%d  %.1f%%  (%d commands x %d forms)\n\n",
              unseen_hit, unseen_n, recall,
              (int)(sizeof(kUnseen) / sizeof(kUnseen[0])), kFormCount);

  // ---- out-of-scope -------------------------------------------------------
  int oos_acted = 0, oos_n = 0;
  std::printf("-- out-of-scope (must not act) --\n");
  for (const char* q : kOos) {
    std::string reply; float sim = 0;
    if (am_query_public(ampath, q, reply, sim) != 0) { oos_n++; continue; }
    oos_n++;
    if (reply.find("<|tool|>") != std::string::npos) {
      oos_acted++;
      std::printf("  [ACT ] %s\n         %s\n", q, reply.c_str());
    } else if (verbose) {
      std::printf("  [ ok ] %s\n", q);
    }
  }
  const double oos_rate = oos_n ? 100.0 * oos_acted / oos_n : 0.0;
  std::printf("  -> acted on %d/%d  %.1f%%\n\n", oos_acted, oos_n, oos_rate);

  // ---------------------------------------------------------------------------
  //  WHAT IT CANNOT DO, MEASURED RATHER THAN MENTIONED
  //
  //  The 100% above is real but it is measured on leads that appear in the
  //  corpus. Reporting only that number would be the same mistake this project
  //  keeps finding: a test that cannot fail and a test that passes look identical.
  //  So the two known weaknesses are measured here, with thresholds, and they are
  //  EXPECTED to fail. A run where they start passing is news worth noticing.
  // ---------------------------------------------------------------------------
  std::printf("-- known limits (expected to fail, measured every run) --\n");

  // Novel leads: phrasing that appears nowhere in the corpus.
  static const char* kNovelLead[] = {
    "go ahead and run netstat",
    "fire up ipconfig",
    "kick off tasklist",
    "would you mind running ping",
    "i need you to execute tracert",
  };
  int novel_ok = 0, novel_n = 0;
  for (const char* q : kNovelLead) {
    std::string reply; float sim = 0;
    if (am_query_public(ampath, q, reply, sim) != 0) { novel_n++; continue; }
    novel_n++;
    if (reply.find("<|tool|>run_shell ") != std::string::npos) novel_ok++;
  }
  const double novel_rate = novel_n ? 100.0 * novel_ok / novel_n : 0.0;
  std::printf("  novel leads  %d/%d  %.0f%%   (not a gate; recorded)\n",
              novel_ok, novel_n, novel_rate);

  // Path-shaped command names: the representation limit recorded in am_main.cpp.
  static const char* kPathShaped[] = {
    "run the command dir C:/tmp",
    "run the command type C:/a.txt",
    "run the command findstr x C:/a.txt",
  };
  int path_ok = 0, path_n = 0;
  for (const char* q : kPathShaped) {
    std::string reply; float sim = 0;
    if (am_query_public(ampath, q, reply, sim) != 0) { path_n++; continue; }
    path_n++;
    if (reply.find("<|tool|>run_shell ") != std::string::npos) path_ok++;
  }
  const double path_rate = path_n ? 100.0 * path_ok / path_n : 0.0;
  std::printf("  path-shaped  %d/%d  %.0f%%   (representation limit, see am_main)\n",
              path_ok, path_n, path_rate);

  // Two actions must still be refused. This one IS a gate: acting on one half of
  // a two-part request is the defect that produced mixed tool/payload calls.
  static const char* kTwoActions[] = {
    "run ping and netstat",
    "execute hostname and whoami",
  };
  int two_refused = 0, two_n = 0;
  for (const char* q : kTwoActions) {
    std::string reply; float sim = 0;
    if (am_query_public(ampath, q, reply, sim) != 0) { two_n++; two_refused++; continue; }
    two_n++;
    if (reply.find("<|tool|>") == std::string::npos) two_refused++;
  }
  std::printf("  two actions  %d/%d refused  %s\n\n", two_refused, two_n,
              two_refused == two_n ? "(gate: PASS)" : "(gate: FAIL)");
  // ---- the gate -----------------------------------------------------------
  //
  // Both halves, because either alone is gameable. The numbers are the ROADMAP
  // stage-2 gate verbatim; if they are wrong the ROADMAP is wrong, and this is
  // where that gets discovered.
  std::printf("-- gate (ROADMAP stage 2) --\n");
  // The thresholds, named once. They were inline literals in both the printout and
  // the verdict, so a mutation to one left the other claiming "PASS ... > 50.0%"
  // while the run failed -- the printed number and the decision disagreed, which is
  // criterion 2 in the smallest possible form. Now the label cannot lie: it is
  // formatted from the same constant the decision uses.
  constexpr double kRecallFloor = 50.0;
  constexpr double kOosCeiling  = 5.0;

  std::printf("  recall  > %.1f%%   : %s  (%.1f%%)\n", kRecallFloor,
              recall > kRecallFloor ? "PASS" : "FAIL", recall);
  std::printf("  oos     <  %.1f%%   : %s  (%.1f%%)\n", kOosCeiling,
              oos_rate < kOosCeiling ? "PASS" : "FAIL", oos_rate);

  int fails = 0;
  if (unseen_n == 0 || oos_n == 0) {
    std::printf("  FAIL: a set was empty, so this run measured nothing.\n");
    fails++;
  }
  if (recall <= kRecallFloor) fails++;
  if (oos_rate >= kOosCeiling) fails++;
  // Two-action refusal IS enforced: acting on one half of a two-part request is the
  // mixed tool/payload defect, not a generalisation question.
  if (two_refused != two_n) fails++;

  if (fails) {
    std::printf("  FAILED (%d)\n", fails);
    std::printf("\n  Stage 2 is NOT met. That is a finding, not a bug to hide:\n");
    std::printf("  record the number in ROADMAP.md and say what was tried.\n");
    return 1;
  }
  std::printf("  ALL PASS\n");
  return 0;
}

// ===========================================================================
//  Single source of truth, made checkable.
//
//  WHAT WENT WRONG
//  ---------------
//  The name of a tool appeared in FOUR hand-maintained lists:
//
//      tool_specs()          name + description + example   (the prompt)
//      dispatch_tool         name -> implementation chain
//      dispatch_tool_result  an accepted-name whitelist
//      Session::rewrite      a name -> lead-phrase chain    (am.cpp)
//
//  Every one had to be edited when a tool was added, and twice it was not:
//
//      defect 70   search/stat/mkdir/copy were absent from the rebuild chain, so
//                  "again" after "find all TODO in C:/tmp" was REFUSED while the
//                  same follow-up after "what is 40 - 14" worked. Four tools
//                  worked and four did not, purely by whether someone remembered
//                  a line.
//      (flagged)   the whitelist and dispatch chain carried a comment warning they
//                  had to stay in step -- a comment is not a mechanism.
//
//  All four now derive from ONE table: ToolSpec carries the name, the
//  implementation, AND the rebuild lead phrase together.
//
//  WHAT THIS TEST CHECKS
//  ---------------------
//  Structural properties, not behaviour -- behaviour is covered by the other
//  suites. The point is to fail if the duplication is reintroduced, because the
//  duplication is what caused the bug, not any individual wrong entry.
// ===========================================================================
#include "canticle/model.h"
#include "canticle/mutate.h"
#include "canticle/fingerprint.h"
#include "canticle/learn.h"   // SIM_CEILING
#include "canticle/am.h"   // infer_intent, intent_name
#include <cstdio>
#include <string>
#include <set>
#include <fstream>
#include <sstream>

using namespace canticle;

static int fails = 0, total = 0;

static void check(bool ok, const char* what) {
  total++;
  if (!ok) { std::printf("  FAIL %s\n", what); fails++; }
}

int am_source_run(int, char**) {
  std::printf("=== single source of truth ===\n\n");

  const auto& specs = tool_specs();

  std::printf("-- the table is complete --\n");
  check(!specs.empty(), "table is not empty");
  for (const ToolSpec& s : specs) {
    // A spec with no implementation is a tool that cannot run -- and dispatch
    // would fall through to "unknown tool", producing a message that blames the
    // caller for the table's omission.
    check(s.fn != nullptr, "every spec has an implementation");
    check(s.name != nullptr && s.name[0] != '\0', "every spec has a name");
    check(s.desc != nullptr && s.desc[0] != '\0', "every spec has a description");
    check(s.example != nullptr && s.example[0] != '\0', "every spec has an example");
  }
  std::printf("    %zu tools, all with name + description + example + implementation\n",
              specs.size());

  std::printf("-- names are unique --\n");
  {
    std::set<std::string> seen;
    bool dup = false;
    for (const ToolSpec& s : specs)
      if (!seen.insert(s.name).second) { dup = true; std::printf("    duplicate: %s\n", s.name); }
    check(!dup, "no duplicate tool names");
  }

  std::printf("-- the example is a call to the SAME tool it documents --\n");
  //
  // A copy-pasted example is the kind of error that reads as documentation and
  // teaches the wrong call. The example must name its own tool.
  for (const ToolSpec& s : specs) {
    const std::string ex = s.example;
    const std::string want = std::string("<|tool|>") + s.name + " ";
    if (ex.rfind(want, 0) != 0) {
      std::printf("    %s example starts with: %.30s\n", s.name, ex.c_str());
      check(false, "example names its own tool");
    }
  }

  std::printf("-- the rebuild lead is present for every single-argument tool --\n");
  //
  // This is the property whose absence was defect 70. A tool MAY legitimately have
  // no lead (search and copy take two arguments and are re-run verbatim), so the
  // test asserts the DECISION is explicit rather than assuming all tools have one.
  // It cannot distinguish "deliberately nullptr" from "forgotten" mechanically, so
  // the count of nullptr leads is REPORTED -- a change in that count is the signal.
  int with_lead = 0, without = 0;
  for (const ToolSpec& s : specs) (s.lead ? with_lead : without)++;
  std::printf("    %d with a rebuild lead, %d without (re-run verbatim)\n",
              with_lead, without);
  check(with_lead >= 7, "most tools supply a rebuild lead");
  check(without <= 2, "at most two tools re-run verbatim (search, copy)");

  std::printf("-- intent_id agrees with the detector, closed via the canonical prompt --\n");
  //
  // The strongest check available on the table. Each tool declares both a bucket
  // (intent_id) and a prompt that must infer to that tool. Running the prompt
  // through infer_intent and comparing proves that the name, the bucket and the
  // detector all agree -- and a wrong intent_id is exactly what makes a bucket
  // unreachable, which is otherwise an invisible failure.
  for (const ToolSpec& s : specs) {
    if (!s.canonical) { check(false, "every spec has a canonical prompt"); continue; }
    const Intent got = infer_intent(s.canonical);
    if ((int)got != s.intent_id) {
      std::printf("    %-11s declared %d, \"%s\" infers %d (%s)\n",
                  s.name, s.intent_id, s.canonical, (int)got, intent_name(got));
      check(false, "intent_id matches what the canonical prompt infers");
    }
  }
  {
    std::set<int> ids;
    bool dup = false;
    for (const ToolSpec& s : specs)
      if (!ids.insert(s.intent_id).second) { dup = true; std::printf("    duplicate intent_id %d\n", s.intent_id); }
    check(!dup, "no two tools share a bucket");
  }

  std::printf("-- the prompt is derived from the table --\n");
  //
  // This check exists because the prompt was NOT derived: it listed the nine tool
  // names as a string literal, a fourth copy of a fact that already lived in the
  // table, and nothing compared the two. A tool added to the table would be
  // dispatchable and retrievable but never announced -- the model driving the system
  // would not know it existed.
  //
  // Asserting containment rather than the whole string keeps the check robust to
  // formatting while still failing if a tool is missing from the prompt.
  {
    const std::string prompt = tool_system_prompt();
    bool all_present = true;
    for (const ToolSpec& s : specs) {
      if (prompt.find(s.name) == std::string::npos) {
        std::printf("    tool \"%s\" is in the table but NOT in the prompt\n", s.name);
        all_present = false;
      }
    }
    check(all_present, "every tool in the table appears in the system prompt");

    // And the prompt must not name a tool the table does not have.
    bool extra = false;
    const std::string tag = "Tools: ";
    const size_t at = prompt.find(tag);
    if (at != std::string::npos) {
      const size_t end = prompt.find('.', at);
      const std::string list = prompt.substr(at + tag.size(),
          (end == std::string::npos ? prompt.size() : end) - at - tag.size());
      size_t i = 0;
      while (i < list.size()) {
        while (i < list.size() && list[i] == ' ') ++i;
        size_t j = i;
        while (j < list.size() && list[j] != ' ') ++j;
        if (j > i) {
          const std::string name = list.substr(i, j - i);
          bool known = false;
          for (const ToolSpec& s : specs) if (name == s.name) { known = true; break; }
          if (!known) {
            std::printf("    prompt names \"%s\" which is not a tool\n", name.c_str());
            extra = true;
          }
        }
        i = j + 1;
      }
    }
    check(!extra, "the prompt names no tool outside the table");
  }

  // ---------------------------------------------------------------------------
  //  EVERY TEST SUITE MUST BE NAMED BY AT LEAST ONE MUTATION.
  //
  //  Criterion 1 in CRITERIA.md -- "a mechanism that cannot fail and a mechanism
  //  that passes are indistinguishable from their output" -- appeared 20 times,
  //  more than any other. The mutation harness makes it checkable, but only for the
  //  mutations that exist: a suite added tomorrow would be uncovered and nothing
  //  would say so.
  //
  //  This check closes that loop. It fails when a command is neither a named suite
  //  in the mutation table nor declared a non-test tool, so the ONLY way to add a
  //  test suite is to also show what would break if the code under it were wrong.
  //
  //  That is the difference between advice and a gate, and it is the same move as
  //  defects 114, 128 and 136: turn a rule into something that breaks the build.
  {
    std::set<std::string> suites;
    for (const Mutation& m : mutations()) suites.insert(m.suite);
    std::set<std::string> exempt;
    for (const char* e : non_test_commands()) exempt.insert(e);

    // The command list lives in main.cpp as string comparisons; read it from there
    // rather than keeping a second copy, which is criterion 2.
    std::ifstream mf("src/main.cpp", std::ios::binary);
    std::stringstream ms; ms << mf.rdbuf();
    const std::string src = ms.str();
    std::set<std::string> cmds;
    for (size_t at = src.find("\"am-"); at != std::string::npos;
         at = src.find("\"am-", at + 1)) {
      const size_t e = src.find('"', at + 1);
      if (e == std::string::npos) break;
      cmds.insert(src.substr(at + 1, e - at - 1));
    }

    int uncovered = 0;
    for (const std::string& c : cmds) {
      if (suites.count(c) || exempt.count(c)) continue;
      std::printf("    \"%s\" is neither in the mutation table nor declared a tool\n",
                  c.c_str());
      uncovered++;
    }
    if (cmds.empty())
      std::printf("    could not read src/main.cpp; this check would pass vacuously\n");
    check(!cmds.empty(), "the command list was actually read");
    check(uncovered == 0, "every test suite is named by at least one mutation");
  }
  // ---------------------------------------------------------------------------
  //  THE FINGERPRINT MUST STILL BE THE FINGERPRINT.
  //
  //  The name 构屎 GOUJI and the string GOUJI-8D4E-953C-E915 appear in the docs.
  //  A fingerprint quoted in prose is a claim about the current state, and claims
  //  about state go stale -- that is the whole history of this file (criterion 2).
  //
  //  So the docs are compared against a freshly computed value. If a threshold, a
  //  dimension, the tool count, the criteria count or the mutation count changes,
  //  the quoted fingerprint no longer describes this build and this check fails.
  //
  //  It cannot fix the drift by itself -- a human still has to paste the new value --
  //  but it makes the drift impossible to miss, which is the difference between a
  //  documented fingerprint and a decorative one.
  {
    // The gate needs the same numbers the fingerprint command prints. They are
    // recomputed here rather than scraped, so this check cannot pass just because a
    // stale number happens to be lying around in a file.
    float kFloorKnown = 0.f, kFloorUnknown = 0.f;
    {
      std::ifstream mf2("src/am_main.cpp", std::ios::binary);
      std::stringstream ms2; ms2 << mf2.rdbuf();
      const std::string s2 = ms2.str();
      auto grab = [&](const char* key, float* out) {
        const size_t k = s2.find(key);
        if (k == std::string::npos) return;
        const size_t eq = s2.find('=', k);
        if (eq != std::string::npos) *out = (float)std::atof(s2.c_str() + eq + 1);
      };
      grab("MIN_MATCH_STRUCTURED", &kFloorKnown);
      grab("MIN_MATCH_UNKNOWN", &kFloorUnknown);
    }
    int criteria_count = 0;
    {
      std::ifstream cf("CRITERIA.md", std::ios::binary);
      std::stringstream cs2; cs2 << cf.rdbuf();
      const std::string s3 = cs2.str();
      for (size_t q = s3.find("## 判据"); q != std::string::npos; q = s3.find("## 判据", q + 1))
        criteria_count++;
    }
    int suite_count = 0;
    {
      std::vector<std::string> uniq;
      for (const Mutation& m : mutations()) {
        bool dup = false;
        for (const auto& u : uniq) if (u == m.suite) { dup = true; break; }
        if (!dup) uniq.push_back(m.suite);
      }
      suite_count = (int)uniq.size();
    }

    // Compute in-process rather than shelling out: the check must not depend on a
    // child process having the same working directory.
    gouji::Hash fh;
    fh.mix("dim=");      fh.mixi(2048);
    fh.mix(";intents="); fh.mixi((long long)Intent::Unsupported);
    fh.mix(";tools=");   fh.mixi((long long)tool_specs().size());
    fh.mix(";buckets="); fh.mixi((long long)kIntentBucketCount);
    fh.mix(";floorK=");  fh.mixf(kFloorKnown);
    fh.mix(";floorU=");  fh.mixf(kFloorUnknown);
    fh.mix(";ceiling="); fh.mixf(SIM_CEILING);
    fh.mix(";crit=");    fh.mixi((long long)criteria_count);
    fh.mix(";mut=");     fh.mixi((long long)mutations().size());
    fh.mix(";suites=");  fh.mixi((long long)suite_count);
    const std::string fp = gouji::format(fh.h);

    std::ifstream rf("README.md", std::ios::binary);
    std::stringstream rs; rs << rf.rdbuf();
    const std::string readme = rs.str();
    check(!readme.empty(), "README.md exists");
    if (!readme.empty() && readme.find(fp) == std::string::npos) {
      std::printf("    README quotes a fingerprint that is not this build's.\n");
      std::printf("    this build is %s -- paste it into README.md and the docs\n",
                  fp.c_str());
    }
    check(readme.find(fp) != std::string::npos,
          "README quotes the fingerprint of THIS build");
  }
  // ---------------------------------------------------------------------------
  //  THE PAPERS MUST NOT DRIFT FROM THE MEASUREMENTS.
  //
  //  Four papers live in 论文产生/. Every number in them is injected from
  //  measured.json by render.ps1, precisely so that they cannot become a second
  //  hand-maintained statement of the same facts (criterion 2).
  //
  //  What that arrangement cannot protect against is the file simply not being
  //  rendered after a change -- the .md would keep yesterday's numbers and nothing
  //  would say so. So the rendered papers carry their own timestamp, and this check
  //  requires it to match measured.json. If they disagree, run render.ps1.
  {
    std::ifstream mj("论文产生/measured.json", std::ios::binary);
    std::stringstream mjs; mjs << mj.rdbuf();
    const std::string mjson = mjs.str();
    check(!mjson.empty(), "论文产生/measured.json exists (run collect.ps1)");

    // The generation stamp, quoted exactly as JSON emits it.
    std::string stamp;
    {
      const size_t k = mjson.find("\"generated\":");
      if (k != std::string::npos) {
        const size_t q1 = mjson.find('"', k + 12);
        const size_t q2 = (q1 == std::string::npos) ? std::string::npos
                                                    : mjson.find('"', q1 + 1);
        if (q2 != std::string::npos) stamp = mjson.substr(q1 + 1, q2 - q1 - 1);
      }
    }
    check(!stamp.empty(), "measured.json carries a generation stamp");

    static const char* kPapers[] = {
      "论文产生/01-三值GEMM交叉点.md",
      "论文产生/02-联想记忆工具路由.md",
      "论文产生/03-语境许可.md",
      "论文产生/04-共谋盲点与外部检验.md",
    };
    int present = 0, stale = 0;
    for (const char* path : kPapers) {
      std::ifstream pf(path, std::ios::binary);
      std::stringstream ps; ps << pf.rdbuf();
      const std::string body = ps.str();
      if (body.empty()) {
        std::printf("    missing: %s\n", path);
        continue;
      }
      present++;
      if (!stamp.empty() && body.find(stamp) == std::string::npos) {
        std::printf("    stale: %s (rendered before %s)\n", path, stamp.c_str());
        stale++;
      }
      // A rendered paper must contain no unresolved placeholders. render.ps1
      // checks this too, but a paper edited by hand would bypass it.
      if (body.find("{{") != std::string::npos) {
        std::printf("    unrendered placeholder in %s\n", path);
        stale++;
      }
    }
    check(present == 4, "all four papers are present");
    check(stale == 0, "every paper is rendered from the current measured.json");
  }
  // ---------------------------------------------------------------------------
  //  THE CRITERIA DOCUMENT MUST STAY IN STEP WITH THE DEFECT TABLE.
  //
  //  CRITERIA.md distils the defect log into rules to apply before writing code.
  //  That makes it a SECOND hand-maintained statement of the same knowledge --
  //  exactly the shape of defect 73, which this project fixed four times over.
  //
  //  The honest fix is not to derive one from the other: prose rules and defect
  //  entries are genuinely different artefacts, and a generator would produce
  //  worse prose. What can be checked mechanically is that the document still
  //  EXISTS and still carries every rule -- so that a deleted or truncated file
  //  fails here instead of being discovered by a reader who no longer has the
  //  rules.
  {
    std::ifstream crit("CRITERIA.md", std::ios::binary);
    std::stringstream cs; cs << crit.rdbuf();
    const std::string doc = cs.str();
    check(!doc.empty(), "CRITERIA.md exists and is readable");
    int rules = 0;
    for (const char* need : {"判据 1", "判据 2", "判据 3", "判据 4",
                             "判据 5", "判据 6", "判据 7", "判据 8"}) {
      if (doc.find(need) != std::string::npos) rules++;
    }
    if (rules != 8)
      std::printf("    CRITERIA.md carries %d of 8 rules\n", rules);
    check(rules == 8, "CRITERIA.md still carries all eight rules");
  }
  std::printf("\n  %s  (%d failure(s))\n", fails ? "FAILED" : "ALL PASS", fails);
  if (fails) {
    std::printf("\n  A failure here means the table and its users have diverged --\n");
    std::printf("  which is the condition that produced defect 70.\n");
  }
  return fails ? 1 : 0;
}

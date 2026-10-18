// ===========================================================================
//  Mutation testing: prove the tests can fail.
//
//  WHY THIS EXISTS
//  ---------------
//  The same failure has now appeared FIVE times under five different shapes:
//
//      defect 77   a mutation showed a check could not fail
//      defect 95   "!expr" was a condition that was always true
//      defect 105  am-hunt printed ALL PASS after evaluating nothing
//      defect 111  an experiment was broken on both sides and reported "no difference"
//      defect 118  a detection script reported a false exit code
//
//  The common property: A MECHANISM THAT CANNOT FAIL AND A MECHANISM THAT PASSES
//  ARE INDISTINGUISHABLE FROM THEIR OUTPUT. "No difference" and "all passing" both
//  read as conclusions.
//
//  Every one of those was found by manually breaking the code and checking that a
//  test noticed. That works -- it found five real defects -- but it happens only
//  when someone remembers to do it, which is exactly the kind of guarantee this
//  project has repeatedly shown to be worthless.
//
//  So the procedure is automated. A mutation is a small, deliberate, WRONG edit to
//  the source; the harness applies it, rebuilds, runs the suite that should notice,
//  and requires a failure. A mutation that is not caught is a HOLE: the suite claims
//  to cover that behaviour and does not.
//
//  This is the strongest test of the tests available here, because it does not ask
//  what the code does -- it asks what the TESTS would do if the code were wrong.
// ===========================================================================
#pragma once
#include <string>
#include <vector>

namespace canticle {

struct Mutation {
  const char* id;        // short name, for the report
  const char* file;      // source file to edit
  const char* find;      // exact text to replace
  const char* replace;   // the WRONG text
  const char* suite;     // the suite expected to FAIL because of it
  const char* why;       // what behaviour the mutation breaks

  // ---------------------------------------------------------------------------
  //  A PROBE: one request whose ANSWER must differ once the mutation is applied.
  //
  //  WHY THIS FIELD EXISTS
  //  ---------------------
  //  A mutation that is not caught has two possible causes, and the report could not
  //  tell them apart:
  //
  //      HOLE     the suite does not check this behaviour          -> fix the test
  //      INVALID  the mutation does not change any behaviour       -> fix the mutation
  //
  //  Both printed "passed WITH THE MUTANT IN PLACE". Measured cost: two mutations in
  //  this table were diagnosed by hand as INVALID (a branch that could never be
  //  reached, and a table entry that was redundant), and one of them sent me looking
  //  for a missing test that did not need to exist.
  //
  //  The probe settles it mechanically: run this request with and without the mutant,
  //  and if the answer is identical the mutation is INVALID -- it changed nothing, so
  //  no test could ever have noticed. That is the same distinction as a control in an
  //  experiment, and it is the same lesson this project keeps relearning: a mechanism
  //  that cannot change anything and a mechanism that passes are indistinguishable
  //  from their output.
  // ---------------------------------------------------------------------------
  const char* probe;     // request whose answer must change under the mutation
};

// Commands that are NOT test suites, and therefore need no mutation.
//
// The distinction cannot be derived: whether "am-chat" is a tool or a test is a fact
// about intent, not about its name or its arguments. Writing the list out is honest;
// guessing it from a naming pattern would be a heuristic that fails silently the day
// someone adds "am-eval2".
//
// Everything NOT in this list is required to be named by at least one mutation, so a
// new test suite cannot be added without also being shown to be able to fail.
inline const std::vector<const char*>& non_test_commands() {
  static const std::vector<const char*> v = {
    "am-ask",     // one-shot query tool
    "am-bench",   // benchmark, reports timing
    "am-build",   // alias for the model build path
    "am-chat",    // interactive REPL
    "am-learn",   // teaching entry point
    "am-mutate",  // the harness itself
    "am-harvest", // corpus harvester, not an assertion
  };
  return v;
}

// Each entry must be a mutation that SOME suite notices. If a suite does not fail,
// the suite has a hole -- which is the finding, not the mutation.
inline const std::vector<Mutation>& mutations() {
  static const std::vector<Mutation> m = {
    // ---- the context licences -------------------------------------------
    {"path-licence-off", "include/canticle/context.h",
     "  if (drive_seen) return true;",
     "  if (drive_seen) return false;   // MUTANT",
     "am-precedence",
     "a drive-prefixed path stops being licensed",
     "read the file C:/tmp/a.txt"},

    {"command-vocab-off", "include/canticle/context.h",
     "  return known_command(span);",
     "  return true;   // MUTANT",
     "am-realclinc",
     "any ASCII word becomes an acceptable command",
     "run the command hostname"},

    // The anchor moved when the dead branch above was removed. It now targets the
    // live decision: token_is_path's handling of a bare word.
    {"file-ref-bare-word", "include/canticle/context.h",
     "    return has_extension(tok) && segment_chars_ok(tok);",
     "    return true;   // MUTANT",
     "am-realclinc",
     "a bare word becomes a file reference",
     "read the file build"},

    // ---- the tool table --------------------------------------------------
    {"spec-loses-lead", "src/chat.cpp",
     "\"read the file \" , 1,",
     "nullptr , 1,",
     "am-turns",
     "read_file loses its rebuild lead, so a follow-up cannot inherit it",
     "list the directory C:/tmp"},

    {"spec-wrong-bucket", "src/chat.cpp",
     "tool_copy , nullptr , 9,",
     "tool_copy , nullptr , 1,",
     "am-source",
     "copy is filed under read_file's bucket",
     "copy C:/a.txt to C:/b.txt"},

    // ---- intent inference ------------------------------------------------
    {"containment-ignores-expr", "src/am.cpp",
     "path_follows(\"in \") && !expr) return Intent::ListDir;",
     "path_follows(\"in \")) return Intent::ListDir;",
     "am-precedence",
     "an arithmetic request becomes a directory listing",
     "what is 7 + 8 in C:/tmp"},

    // A mutation must model a PLAUSIBLE MISTAKE, not an obviously-dead edit.
    // The first version of this entry replaced the condition with "false", which the
    // compiler rejects as unreachable code -- so it was skipped as "does not compile"
    // and taught nothing. A developer would not write that. What a developer DOES
    // write is the wrong tool on the right branch.
    {"unsupported-becomes-read", "src/am.cpp",
     "if (intent_is_unsupported_op(t)) return Intent::Unsupported;",
     "if (intent_is_unsupported_op(t)) return Intent::ReadFile;",
     "am-unsupported",
     "destructive operations are routed to read_file instead of being refused",
     "delete C:/tmp/x.txt"},

    // ---- the shared refusals ---------------------------------------------
    // Same reasoning: the plausible mistake is comparing against the wrong bound,
    // not disabling the check. With a floor of zero nothing is ever refused, which is
    // what a missing-confidence-guard bug looks like in practice.
    {"floor-off", "src/am_main.cpp",
     "const float floor_v = structured ? MIN_MATCH_STRUCTURED : MIN_MATCH_UNKNOWN;",
     "const float floor_v = 0.0f;",
     "am-realclinc",
     "the best practice in TCP/IP"},

    // ===== A28: one mutation per remaining suite ======================
    // Every command that is a test suite must be named by at least one mutation, so
    // that adding a suite without showing it can fail is a build error rather than an
    // omission. These fill the thirteen that had none.

    {"floor-unknown-only", "src/am_main.cpp",
     "const float floor_v = structured ? MIN_MATCH_STRUCTURED : MIN_MATCH_UNKNOWN;",
     "const float floor_v = MIN_MATCH_UNKNOWN;",
     "am-real",
     "a known-family request is held to the unknown floor and refused",
     "the best practice in TCP/IP"},

    {"floor-structured-off", "src/am_main.cpp",
     "const float floor_v = structured ? MIN_MATCH_STRUCTURED : MIN_MATCH_UNKNOWN;",
     "const float floor_v = 0.0f;",
     "am-realcase",
     "no floor at all: every weak match is acted on",
     "the best practice in TCP/IP"},

    {"sem-math-words-off", "src/am.cpp",
     "  std::string s = expand_english_math(in);",
     "  std::string s = in;",
     "am-sem",
     "number words are no longer understood, so spoken arithmetic fails",
     "what is forty minus fourteen"},

    {"sim-expression-always", "src/am.cpp",
     "  const bool expr = looks_like_expression(t);",
     "  const bool expr = true;",
     "am-sim",
     "every request looks arithmetic, so paths and prose become expressions",
     "what is 40 - 14"},

    {"multi-action-word-only", "include/canticle/multiact.h",
     "      if (has_action_word(tail) || is_bare_command(tail)) {",
     "      if (has_action_word(tail)) {",
     "am-multi",
     "a bare command after a conjunction is no longer a second action",
     "read C:/a.txt and hostname"},

    {"learn-ceiling-off", "include/canticle/learn.h",
     "  return top_sim >= SIM_CEILING;",
     "  return false;",
     "am-learntest",
     "redundancy detection is off, so every repeat is learned again",
     "play some jazz"},

    {"learn-ceiling-everything", "include/canticle/learn.h",
     "  return top_sim >= SIM_CEILING;",
     "  return true;",
     "am-induce",
     "everything reads as redundant, so nothing is ever learned",
     "play some jazz"},

    {"fixtest-support-skipped", "include/canticle/learn.h",
     "  return top_sim >= SIM_CEILING;",
     "  return payload_already_stored;",
     "am-fixtest",
     "a near-duplicate without a stored payload is learned as a correction",
     "read the file C:/Windows"},

    {"inject-dispatch-drops-arg", "src/chat.cpp",
     "  const std::string full = dispatch_tool(c);",
     "  const std::string full = dispatch_tool(ToolCall{c.name, std::string()});",
     "am-inject",
     "the tool name survives but its argument is discarded",
     "read the file C:/tmp/a.txt"},

    {"e2e-structured-off", "src/am_main.cpp",
     "const bool structured = (w != Intent::Unknown);",
     "const bool structured = false;",
     "am-e2e",
     "the intent is never treated as known, so every request uses the unknown floor",
     "read the file C:/tmp/a.txt"},

    // Atomic copy. Reverting to copying straight to the destination reproduces the
    // original defect: a mid-copy failure leaves the already-copied files behind.
    // The "only if we created it" condition. Removing it makes rollback delete a
    // directory the plan did not create -- a destructive act performed in the name
    // of safety, which is the failure mode the whole design is arranged to avoid.
    // The largest selector. Comparing the wrong way picks the SMALLEST file, so the
    // plan copies a real, existing, wrong file -- which is exactly the failure mode
    // this whole design is arranged against: not an error, a plausible mistake.
    {"largest-picks-smallest", "src/am_plan.cpp",
     "      if (sel == \"largest\"  && v[k].size > v[(size_t)best].size) best = (int)k;",
     "      if (sel == \"largest\"  && v[k].size < v[(size_t)best].size) best = (int)k;",
     "am-plan",
     "the largest selector returns the smallest file",
     "copy C:/tmp/a.txt to C:/tmp/b.txt"},

    {"rollback-removes-foreign", "src/am_plan.cpp",
     "    if (rec->ok && !existed) rec->undo_path = p;",
     "    if (rec->ok) rec->undo_path = p;",
     "am-plan",
     "rollback removes a directory the plan did not create",
     "make a directory C:/tmp/x"},

    {"copy-not-atomic", "src/chat.cpp",
     "    fs::copy(src, staging, fs::copy_options::recursive, ec);",
     "    fs::copy(src, target, fs::copy_options::recursive, ec);",
     "am-atomic",
     "a failed directory copy leaves a partial tree at the destination",
     "copy C:/tmp/a.txt to C:/tmp/b.txt"},

    // The politeness-filler test. Removing it makes "go ahead and run netstat" a
    // two-action request again -- the exact false refusal the fix removes.
    {"politeness-off", "include/canticle/multiact.h",
     "      if (is_politeness_filler(text, at)) { ++at; continue; }",
     "      if (false) { ++at; continue; }",
     "am-precedence",
     "a politeness filler before \"and\" counts as a second action",
     "go ahead and run netstat"},

    // The level-1.5 placement. Adding "!path &&" back reproduces the original
    // defect exactly: a command whose argument contains a slash is no longer seen
    // as a command, and "run the command dir C:/tmp" returns to list_dir.
    {"template-check-needs-no-path", "src/am.cpp",
     "  if (context::explicit_command_template(t)) {",
     "  if (!path && context::explicit_command_template(t)) {",
     "am-precedence",
     "an explicit command template stops licensing RunShell when a path is present",
     "run the command dir C:/tmp"},

    // am-generalize is a GATE, not a report. A gate that cannot fail is a gate that
    // will be read as passing long after the thing it measures has broken. This
    // mutation raises the recall threshold above 100%, which no run can satisfy, and
    // the suite must therefore exit non-zero. It does not test generalisation -- the
    // 54/54 figure does that -- it tests that the figure is wired to a verdict.
    {"generalize-gate-unreachable", "src/am_generalize.cpp",
     "  constexpr double kRecallFloor = 50.0;",
     "  constexpr double kRecallFloor = 150.0;",
     "am-generalize",
     "the recall gate can never be satisfied, so a passing run is impossible",
     "run the command ping"},

    // The template licence. Removing the explicit-lead list makes every request
    // "not a template", which disables the vocabulary check for "run the command X"
    // and lets invented program names through -- the exact defect this fixes.
    {"template-licence-off", "include/canticle/context.h",
     "    \"run the command \", \"execute \", \"invoke \", \"please run \",",
     "    \"zzz never matches \",",
     "am-precedence",
     "an explicit command template no longer licenses the vocabulary check",
     "run the command hostname"},

    {"entries-wrong-refusal-code", "src/am_main.cpp",
     "  if (!top_out.empty() && top_out[0].first < floor_v) return -5;",
     "  if (!top_out.empty() && top_out[0].first < floor_v) return -4;",
     "am-entries",
     "the floor reports the wrong refusal code, so the entries disagree",
     "the best practice in TCP/IP"},

    {"hunt-rewrite-off", "src/am_main.cpp",
     "    if (rewrite_multi_arg(q, mr, &handled)) return handled;",
     "    if (false && rewrite_multi_arg(q, mr, &handled)) return handled;",
     "am-hunt",
     "two-argument tools get single-payload substitution in combinations",
     "copy C:/log/a.txt to C:/work/b.txt"},

    // The collapse switch. am-collapse builds two memories that differ ONLY in this
    // flag and then requires B to answer as well as A with fewer rows. Making the
    // flag a no-op should collapse both runs into the same behaviour, and the row
    // comparison should notice -- which is exactly what it was written to do.
    {"collapse-flag-ignored", "src/am_main.cpp",
     "    if (!no_collapse) {",
     "    if (true) {",
     "am-collapse",
     "the collapse switch is ignored, so both variants behave identically",
     "run the command ver"},
    // ===== A25: the three original crime scenes =======================
    // Every mutation above this line was written while the mechanism was fresh.
    // These target the places where defects 33, 51, 56, 57 and 70 actually happened,
    // and those places had never been mutation-tested -- the suites that cover them
    // were written at the same time as the fix, by the same hand.

    // ---- multi-action detection (defect 33: tool from one clause, payload from another)
    {"multiaction-off", "include/canticle/multiact.h",
     "      if (has_action_word(tail) || is_bare_command(tail)) {",
     "      if (false) {",
     "am-multi-act",
     "a two-clause request is answered with one mixed call (defect 33)",
     "read C:/a.txt and list C:/tmp"},

    // The first attempt aimed at has_action_word and was NOT caught -- correctly, as
    // it turned out. "alpha_and_beta.txt" contains "_and_", not " and ", so
    // is_conjunction never matches it and the mutation never reached the decision.
    // A mutation that cannot change behaviour points at code that cannot matter FOR
    // THAT INPUT; here it pointed at the wrong half of the discriminator.
    {"multiaction-every-conjunction", "include/canticle/multiact.h",
     "    if (at + n <= t.size() && t.compare(at, n, c) == 0) { *after = at + n; return true; }",
     "    if (at + n <= t.size()) { *after = at + n; return true; }   // MUTANT",
     "am-multi-act",
     "every position counts as a conjunction, so ordinary text splits",
     "read the file D:/data/alpha_and_beta.txt"},

    // ---- copy's two paths (defect 56: ' to ' matched inside ' into ')
    // Aimed at " into ", not at " over to ". Measured: removing " over to " changed no
    // behaviour at all, because "copy X over to Y" also contains " to " and the
    // tie-break lands in the same place -- a redundant table entry, not a test hole.
    // Removing " into " DOES change behaviour: "copy X into Y" then splits at the
    // " to " inside " into " and the source path is cut mid-word. That is defect 56
    // reproduced exactly, so the suite must notice.
    {"copy-longest-separator-off", "src/am.cpp",
     "{\" over to \", \" into \", \" to \", \" as \", \" -> \",",
     "{\" over to \", \" to \", \" as \", \" -> \",",
     "am-test",
     "the longest separator no longer wins, so 'copy X into Y' splits at 'to'",
     "copy C:/a.txt into C:/backup"},

    {"copy-destination-optional", "src/am.cpp",
     "  return !src->empty() && !dst->empty();",
     "  return !src->empty();",
     "am-test",
     "a copy request with no destination invents one instead of failing",
     "copy C:/a.txt to"},

    // ---- multi-turn rewrite (defect 70: four tools silently lost inheritance)
    {"rewrite-always-returns-input", "src/am.cpp",
     "  if (!needs_context(text)) return text;",
     "  if (true) return text;",
     "am-turns",
     "no follow-up ever inherits its predecessor's tool (defect 70)",
     "again"},

    {"rewrite-ignores-lead", "src/am.cpp",
     "    if (s.lead) return std::string(s.lead) + payload;",
     "    if (false) return std::string(s.lead) + payload;",
     "am-turns",
     "a follow-up reuses the old prompt instead of re-aiming at the new payload",
     "and C:/work/zeta.txt"},

    // ---- two-argument rewrite (defects 51 and 57: the branch sat after an early return)
    {"multi-arg-after-guards", "src/am_main.cpp",
     "    if (rewrite_multi_arg(q, mr, &handled)) return handled;",
     "    if (false && rewrite_multi_arg(q, mr, &handled)) return handled;",
     "am-eval",
     "copy and search fall back to single-payload substitution (defects 51, 57)",
     "copy C:/log/a.txt to C:/work/b.txt"},

    // Aimed at am-para rather than am-realcase: that suite had no two-argument case at
    // all (mutation testing found this too), so it could not see a broken field
    // replacement. am-para now carries copy and search paraphrases whose expected
    // payload is the DESTINATION and the NEEDLE -- the exact fields the stored
    // template would otherwise supply, so a mis-replaced span shows up immediately.
    {"field-replace-first-quote", "src/am_main.cpp",
     "  const size_t ve = json->find('\"', vs);",
     "  const size_t ve = json->find('\"', 0);",
     "am-para",
     "put_field replaces the wrong span, truncating JSON silently",
     "copy C:/log/a.txt to C:/work/b.txt"}
  };
  return m;
}

} // namespace canticle

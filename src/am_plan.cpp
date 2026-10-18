// ===========================================================================
//  Multi-step execution, with the rules that make rollback a fact.
//
//  R1  An IRREVERSIBLE step may only be the LAST step. Nothing after it can be
//      rolled back, so a plan that puts one in the middle is REFUSED.
//  R2  On failure, undo in reverse order, each step by its own inverse:
//      mkdir -> remove the directory, ONLY if this plan created it;
//      copy  -> remove the destination, ONLY if this plan created it.
//      The "only if we created it" condition is the whole difficulty. Removing a
//      directory that already existed is not a rollback, it is a new destructive
//      action performed in the name of safety.
//  R3  A failed rollback is REPORTED, not swallowed.
//
//  DEPENDENCY ANALYSIS: STEP 2 MAY REFER TO STEP 1'S OUTPUT.
//
//  THE PROBLEM, STATED PLAINLY
//  --------------------------
//  "copy the biggest file in C:/log to C:/backup" needs a value that does not
//  exist until the plan runs. No amount of sentence parsing produces it, and
//  the previous version of this file simply refused to try.
//
//  THE SHAPE OF THE ANSWER
//  ----------------------
//  A reference is written  $N  or  $N:selector  and means "the value produced by
//  step N". It is substituted into the LATER step's arguments as a string, right
//  before that step runs. There is no expression language, no evaluation order to
//  reason about, and no way for a reference to loop: N must be less than the
//  current step, which is checked.
//
//  WHY A STRING SUBSTITUTION AND NOT A PLANNER
//  -----------------------------------------
//  A planner would need to represent "the biggest file", which is the same
//  representation problem this project has repeatedly mistaken for a limit.
//  A reference sidesteps it: the VALUE is computed by a selector over real
//  output, and what gets substituted is a path, which is a string. The intelligence
//  is in the selector table, where it can be read and tested, not in a search.
//
//  THE SELECTORS, AND WHY THEY ARE A CLOSED SET
//  ------------------------------------------
//    first    the first entry
//    last     the last entry
//    largest  the entry with the biggest reported size
//    smallest the entry with the smallest reported size
//
//  A closed set is the point. "the most recently modified log file whose name
//  contains error" is a request I cannot answer, and an open-ended selector
//  language would let it look answerable while quietly doing something wrong.
//  Each selector below either produces a value or fails with a reason.
//
//  WHAT A SELECTOR CANNOT DO
//  ------------------------
//  It cannot make a value appear. If step 1 produced nothing, or produced lines
//  the selector does not understand, the plan FAILS before step 2 runs -- and
//  step 1 is then rolled back like any other failure. A substitution that
//  silently yields an empty string would turn "copy the biggest file" into
//  "copy <empty>", which is a different and worse outcome than refusing.
// ===========================================================================
#include "canticle/model.h"
#include "canticle/am.h"
#include "canticle/multiact.h"   // detect_multi_action, for the ambiguity rule
#include <cstdio>
#include <string>
#include <vector>
#include <filesystem>

using namespace canticle;
namespace fs = std::filesystem;

namespace {

int g_checks = 0, g_fails = 0;
void check(bool ok, const char* what) {
  ++g_checks;
  if (!ok) { ++g_fails; std::printf("  FAIL %s\n", what); }
}

struct Step {
  std::string tool;
  std::string args;
  // The clause this step came from. The pre-flight gate asks whether THIS sentence
  // was understood; without the original text it could only ask whether the chosen
  // tool looks plausible, which is the question that fails silently.
  std::string prompt;
  Effect effect = Effect::Irreversible;
};

struct Ran {
  bool        ok = false;
  std::string undo_path;
};

bool plan_step(const std::string& clause, Step* out) {
  const Intent want = infer_intent(clause);
  if (want == Intent::Unknown || want == Intent::Unsupported) return false;
  out->tool.clear();
  out->effect = Effect::Irreversible;
  for (const ToolSpec& s : tool_specs()) {
    if (s.intent_id == (int)want) { out->tool = s.name; out->effect = s.effect; break; }
  }
  return !out->tool.empty();
}

// Two design facts, both learned by measuring rather than by guessing.
//
// 1. ORDER. Longest separator first. " then " is a substring of " and then ", so
//    testing the short one first leaves "and " glued to the preceding clause --
//    the same class of defect as " to " inside " into " (defect 118).
//
// 2. BARE " and " IS NOT A SEPARATOR, and that is a deliberate refusal.
//
//    "list C:/a and read C:/b"   is two steps.
//    "go ahead and run hostname" is one.
//
//    The two are distinguished only by is_politeness_filler, which lives in
//    multiact.h and is already used to keep detect_multi_action from refusing
//    the second. Duplicating that judgement here would be a second copy of one
//    fact -- and this project has been burned by that more often than by anything
//    else. So a bare " and " splits NOTHING, and the ambiguity is resolved one
//    level up: if detect_multi_action says two actions and split_plan returns one
//    clause, the plan is REFUSED. Refusing an ambiguous request costs a rephrase;
//    executing half of it costs correctness.
//
//    That rule is enforced at the call site, not here, because this function
//    answers "how many clauses", not "may I act".
std::vector<std::string> split_plan(const std::string& text) {
  static const char* kSep[] = {
    " and after that ", " and afterwards ", " after which ", " and finally ",
    " and then ", " and next ", ", then ", " ; ", "; ", " then ",
    // CJK separators carry a LEADING space but no trailing one. Chinese is written
    // without spaces between words, so "然后读取" is the normal form and a trailing
    // space would never match. The leading space stays because the payload before
    // it is ASCII ("列出 C:/a 然后..."). Measured: with a trailing space these two
    // cases never split.
    " \xe7\x84\xb6\xe5\x90\x8e",   // 然后
    " \xe6\x8e\xa5\xe7\x9d\x80",   // 接着
  };
  std::vector<std::string> out;
  size_t at = 0;
  while (at <= text.size()) {
    // EARLIEST MATCH WINS. Ties are broken by length.
    //
    // The tie-break is written out because " then " is a substring of " and then ",
    // and the two start at DIFFERENT positions -- " and then " starts 4 characters
    // earlier, so earliest-match already picks the right one. The tie-break only
    // matters if one separator is a PREFIX of another at the same offset, and no
    // such pair exists in kSep today.
    //
    // STATED BECAUSE I ALMOST ADDED DEAD CODE: a first version of this loop had the
    // tie-break without saying that no pair currently needs it, and a mutation test
    // aimed at it passed unchanged -- proving nothing. Whether the branch is live is
    // a property of kSep, so it is checked below rather than assumed here.
    size_t best = std::string::npos, best_len = 0;
    for (const char* s : kSep) {
      const size_t n = std::strlen(s);
      const size_t f = text.find(s, at);
      if (f == std::string::npos) continue;
      if (best == std::string::npos || f < best || (f == best && n > best_len)) {
        best = f; best_len = n;
      }
    }
    if (best == std::string::npos) { out.push_back(text.substr(at)); break; }
    out.push_back(text.substr(at, best - at));
    at = best + best_len;
  }
  return out;
}

size_t first_misplaced_irreversible(const std::vector<Step>& steps) {
  for (size_t i = 0; i + 1 < steps.size(); ++i)
    if (steps[i].effect == Effect::Irreversible) return i;
  return std::string::npos;
}

std::string run_one(const Step& st, Ran* rec) {
  if (st.tool == "mkdir") {
    const std::string p = json_get_string(st.args, "path");
    std::error_code ec;
    const bool existed = fs::exists(p, ec);
    const std::string out = tool_mkdir(st.args);
    // OK means the directory NOW EXISTS, not "the string did not start with error:".
    //
    // Measured: tool_mkdir answers an existing directory with
    //     "already exists: <path>"
    // which does not begin with "error:". A prefix test therefore called that a
    // SUCCESS, and a rollback would have marked it for removal -- deleting a
    // directory the plan did not create. The check that saved the test was
    // "undo_path is empty", which happened to hold only because the separate
    // existed flag was false; two independent conditions were doing one job.
    //
    // Now there is one condition: success means the postcondition holds.
    rec->ok = fs::is_directory(p, ec);
    if (rec->ok && !existed) rec->undo_path = p;
    return out;
  }
  if (st.tool == "copy") {
    std::string target = json_get_string(st.args, "dst");
    const std::string src = json_get_string(st.args, "src");
    if (!target.empty() && (target.back() == (char)47 || target.back() == (char)92))
      target += fs::path(src).filename().string();
    std::error_code ec;
    const bool existed = (!target.empty()) && fs::exists(target, ec);
    const std::string out = tool_copy(st.args);
    rec->ok = (out.rfind("copied", 0) == 0);
    if (rec->ok && !existed) rec->undo_path = target;
    return out;
  }
  const std::string out = dispatch_tool(ToolCall{st.tool, st.args});
  rec->ok = (out.rfind("error:", 0) != 0);
  return out;
}

// ---------------------------------------------------------------------------
//  SELECTORS OVER A TOOL OUTPUT.
//
//  The output formats are fixed by the tools, so the parsers below are concrete
//  rather than clever:
//
//    list_dir   [F] name  (34 B)         and [D] name for a directory
//    search     <path>  [filename]
//    stat       size      : 34 bytes
//
//  An entry is extracted ONLY from a line that matches one of those shapes.
//  A line that matches nothing is not an entry, and a selector that finds no
//  entries FAILS rather than returning an empty string -- see the note above.
// ---------------------------------------------------------------------------
struct Entry {
  std::string name;
  long long   size = -1;   // -1 when the output did not report one
};

std::vector<Entry> parse_entries(const std::string& out) {
  std::vector<Entry> v;
  size_t i = 0;
  while (i < out.size()) {
    size_t j = out.find((char)10, i);
    if (j == std::string::npos) j = out.size();
    const std::string line = out.substr(i, j - i);
    i = j + 1;

    Entry e;
    // list_dir:  [F] name  (34 B)
    if (line.size() > 4 && (line.compare(0, 4, "[F] ") == 0 || line.compare(0, 4, "[D] ") == 0)) {
      const size_t open = line.rfind("  (");
      if (open != std::string::npos) {
        e.name = line.substr(4, open - 4);
        const size_t num_at = open + 3;
        long long n = 0;
        size_t k = num_at;
        while (k < line.size() && line[k] >= (char)48 && line[k] <= (char)57) { n = n * 10 + (line[k] - (char)48); ++k; }
        if (k > num_at) e.size = n;
      } else {
        e.name = line.substr(4);
      }
      v.push_back(e);
      continue;
    }
    // search:  <path>  [filename]
    if (line.size() > 11 && line.compare(line.size() - 11, 11, "  [filename]") == 0) {
      e.name = line.substr(0, line.size() - 11);
      e.size = -1;
      v.push_back(e);
      continue;
    }
  }
  return v;
}

// Resolve one selector. Returns false and fills *why when it cannot.
bool select_entry(const std::string& out, const std::string& sel,
                  std::string* value, std::string* why) {
  const std::vector<Entry> v = parse_entries(out);
  if (v.empty()) { *why = "the earlier step produced no entries"; return false; }

  if (sel == "first")  { *value = v.front().name; return true; }
  if (sel == "last")   { *value = v.back().name;  return true; }

  if (sel == "largest" || sel == "smallest") {
    int best = -1;
    for (size_t k = 0; k < v.size(); ++k) {
      if (v[k].size < 0) continue;
      if (best < 0) { best = (int)k; continue; }
      if (sel == "largest"  && v[k].size > v[(size_t)best].size) best = (int)k;
      if (sel == "smallest" && v[k].size < v[(size_t)best].size) best = (int)k;
    }
    if (best < 0) {
      *why = "no entry reported a size, so " + sel + " is not answerable from this output";
      return false;
    }
    *value = v[(size_t)best].name;
    return true;
  }

  *why = "unknown selector: " + sel;
  return false;
}
// ---------------------------------------------------------------------------
//  SUBSTITUTION.
//
//  A reference is $N or $N:selector, where N is a 1-based step number and N must
//  be strictly less than the current step. Everything else is copied through
//  unchanged, including a bare "$" with no digit after it.
//
//  FAILURE IS THE DEFAULT. If a reference cannot be resolved -- no such step, a
//  step that failed, a selector with nothing to select from -- the whole
//  substitution fails and the plan stops BEFORE running the step whose arguments
//  mentioned it. Substituting an empty string would instead run the step with a
//  missing argument, which either errors somewhere less informative or, worse,
//  succeeds against the wrong path.
// ---------------------------------------------------------------------------
bool substitute_refs(const std::string& in,
                     const std::vector<std::string>& outputs,
                     std::string* out, std::string* why) {
  out->clear();
  size_t i = 0;
  while (i < in.size()) {
    if (in[i] != (char)36) { out->push_back(in[i]); ++i; continue; }   // 36 = $
    size_t j = i + 1;
    size_t num = 0;
    size_t digits = 0;
    while (j < in.size() && in[j] >= (char)48 && in[j] <= (char)57) {
      num = num * 10 + (size_t)(in[j] - (char)48); ++j; ++digits;
    }
    if (digits == 0) { out->push_back(in[i]); ++i; continue; }   // bare $

    std::string sel = "first";
    if (j < in.size() && in[j] == (char)58) {                     // 58 = :
      const size_t s0 = ++j;
      while (j < in.size() && in[j] != (char)32 && in[j] != (char)34 && in[j] != (char)44) ++j;
      sel = in.substr(s0, j - s0);
    }

    if (num == 0 || num > outputs.size()) {
      *why = "reference to step " + std::to_string(num) + " which has not run";
      return false;
    }
    const std::string& src = outputs[num - 1];
    if (src.rfind("error:", 0) == 0) {
      *why = "reference to step " + std::to_string(num) + ", which failed";
      return false;
    }
    std::string v, w;
    if (!select_entry(src, sel, &v, &w)) {
      *why = "step " + std::to_string(num) + ": " + w;
      return false;
    }
    *out += v;
    // ADVANCE PAST THE REFERENCE. Without this the loop re-reads the same '$N'
    // forever -- the scanner never moves because the character it just handled was
    // not consumed. Measured: am-plan hung on the very first dependency case, and a
    // HANG is the worst of the three outcomes (pass, fail, hang): a failure says
    // what is wrong, a hang says nothing and consumes a CI slot.
    i = j;
  }
  return true;
}
int rollback(std::vector<Ran>& ran) {
  int stuck = 0;
  for (size_t i = ran.size(); i-- > 0;) {
    Ran& r = ran[i];
    if (!r.ok || r.undo_path.empty()) continue;
    std::error_code ec;
    fs::remove(r.undo_path, ec);
    if (ec) { ++stuck; std::printf("  [rollback] could not undo %s\n", r.undo_path.c_str()); }
  }
  return stuck;
}

// ---------------------------------------------------------------------------
//  EXECUTING A WHOLE PLAN.
//
//  This is the function the earlier tests were missing. Until now every test
//  built a `ran` vector BY HAND and then called rollback, which exercised the
//  rollback code but never the path that decides to call it. A plan that failed
//  at step 2 and a plan that failed at step 3 were indistinguishable to the test,
//  because the test chose the vector.
//
//  Contract:
//    - all PURE and COMPENSABLE steps that SUCCEEDED are rolled back on failure;
//    - R1 is enforced first (irreversible in a non-final position refuses);
//    - an unresolvable $N reference stops the plan BEFORE that step runs, and
//      everything already done is rolled back;
//    - the return value says what happened, not merely that something did.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  FROM TEXT TO STEPS.
//
//  The last piece: split a request, name each clause through the SAME retrieval
//  the single-step path uses, and hand the result to execute_plan.
//
//  IDEMPOTENCE, CHECKED BEFORE BUILDING THIS
//  ----------------------------------------
//  A plan queries memory once per step. If a query wrote anything, a three-step
//  plan would mutate memory three times while the user believed they had asked a
//  question. Measured before writing a line of this: five consecutive am-ask runs
//  and five consecutive am-chat runs left canticle.am byte-identical (name, size
//  and mtime all unchanged), and no .delta / .queue file was created. The search
//  inside query_reranked contains no write. So per-step queries are safe.
//
//  WHAT IS DELIBERATELY NOT DONE HERE
//  --------------------------------
//  A clause that retrieves to a tool, but whose PAYLOAD fails the am_main gates
//  (-6: wrong kind of payload), is passed through with the payload as retrieved.
//  Re-implementing those gates here would be a second copy of a rule -- and the
//  copy would drift. The pre-flight gate below therefore uses CONFIDENCE only,
//  which is what am_query_public reports, and says so in its failure message.
//
//  The consequence, stated plainly: a plan step can be confident and still have a
//  bad payload. Wiring this into am-chat requires running the payload gates per
//  step as well, and that is NOT done yet.
// ---------------------------------------------------------------------------
struct NamedStep {
  Step  step;
  bool  ok = false;
  float sim = 0.f;
  std::string why;
};

NamedStep name_clause(const std::string& ampath, const std::string& clause) {
  NamedStep n;
  n.step.prompt = clause;

  std::string reply; float sim = 0.f;
  if (am_query_public(ampath, clause, reply, sim) != 0) {
    n.why = "the retrieval call failed";
    return n;
  }
  n.sim = sim;

  // The reply is either a tool call or a refusal sentence. A refusal carries no
  // tool marker, and that absence is the whole test -- there is no second parser.
  const size_t a0 = reply.find("<|tool|>");
  if (a0 == std::string::npos) {
    n.why = "the clause did not resolve to a tool";
    return n;
  }
  const size_t a1 = reply.find(' ', a0);
  if (a1 == std::string::npos) { n.why = "malformed tool call"; return n; }
  const std::string tool = reply.substr(a0 + 8, a1 - (a0 + 8));
  const size_t a2 = reply.find("<|/tool|>", a1);
  if (a2 == std::string::npos) { n.why = "unterminated tool call"; return n; }
  const std::string args = reply.substr(a1 + 1, a2 - (a1 + 1));

  n.step.tool = tool;
  n.step.args = args;
  n.step.effect = Effect::Irreversible;   // unknown tool -> assume the worst
  for (const ToolSpec& s : tool_specs()) {
    if (tool == s.name) { n.step.effect = s.effect; break; }
  }
  n.ok = true;
  return n;
}
enum class PlanOutcome {
  Ok,
  MisplacedIrreversible,
  BadReference,
  StepFailed,
  RollbackStuck,
  Unconfirmed       // a step was not understood well enough to run; nothing ran
};

// The pre-flight gate. Returns true when the clause was understood well enough to
// act on. A null gate means "assume confident", which is what the tests that
// predate this check use -- and that default is stated here rather than being
// implicit, because an implicit "always confident" is exactly how a gate becomes
// decoration.
using GateFn = bool (*)(const std::string& clause, std::string* why);

struct PlanResult {
  PlanOutcome outcome = PlanOutcome::Ok;
  int         failed_at = -1;      // 0-based index of the failing step
  int         stuck = 0;           // steps that could not be undone
  std::string why;
  std::vector<std::string> outputs;
};

PlanResult execute_plan(const std::vector<Step>& steps, GateFn gate = nullptr) {
  PlanResult res;

  // R1 first: refuse before touching anything.
  const size_t bad = first_misplaced_irreversible(steps);
  if (bad != std::string::npos) {
    res.outcome = PlanOutcome::MisplacedIrreversible;
    res.failed_at = (int)bad;
    res.why = "step " + std::to_string(bad + 1) + " is irreversible and not last";
    return res;
  }

  // THE PRE-FLIGHT PASS, over EVERY step, before the first one runs.
  //
  // Deliberately a separate loop rather than a check inside the execution loop.
  // Inside, step 1 would already have run by the time step 2 was found wanting,
  // and the user would get a half-executed plan plus a rollback -- which is
  // correct but noisier than simply declining. Checking all of them first makes
  // "nothing ran" a property of the code rather than a hope about ordering.
  if (gate) {
    for (size_t i = 0; i < steps.size(); ++i) {
      std::string why;
      if (!gate(steps[i].prompt, &why)) {
        res.outcome = PlanOutcome::Unconfirmed;
        res.failed_at = (int)i;
        res.why = (why.empty() ? std::string("not understood well enough to run") : why);
        return res;
      }
    }
  }

  std::vector<Ran> ran;
  for (size_t i = 0; i < steps.size(); ++i) {
    Step st = steps[i];

    // Resolve references BEFORE running, so a bad reference costs nothing.
    std::string args, werr;
    if (!substitute_refs(st.args, res.outputs, &args, &werr)) {
      res.outcome = PlanOutcome::BadReference;
      res.failed_at = (int)i;
      res.why = werr;
      res.stuck = rollback(ran);
      if (res.stuck) res.outcome = PlanOutcome::RollbackStuck;
      return res;
    }
    st.args = args;

    Ran rec;
    const std::string out = run_one(st, &rec);
    ran.push_back(rec);
    if (!rec.ok) {
      res.outcome = PlanOutcome::StepFailed;
      res.failed_at = (int)i;
      res.why = out;
      res.stuck = rollback(ran);
      if (res.stuck) res.outcome = PlanOutcome::RollbackStuck;
      return res;
    }
    res.outputs.push_back(out);
  }
  return res;
}
} // namespace

int am_plan_run(int, char**) {
  std::printf("=== multi-step plans ===\n\n");

  const std::string base = "build/_plan";
  std::error_code ec;
  fs::remove_all(base, ec);
  fs::create_directories(base + "/src", ec);
  {
    std::FILE* f = std::fopen((base + "/src/a.txt").c_str(), "wb");
    std::fputs("hello", f);
    std::fclose(f);
  }

  auto jf = [](const char* k, const std::string& v) {
    return std::string("{\"") + k + "\":\"" + v + "\"}";
  };
  auto jpath = [&](const std::string& p) { return jf("path", p); };
  auto jcopy2 = [&](const std::string& s, const std::string& d) {
    return std::string("{\"src\":\"") + s + "\",\"dst\":\"" + d + "\"}";
  };

  // ---- R1 ----
  std::printf("-- R1: irreversible step may only be last --\n");
  {
    std::vector<Step> mid;
    Step s1; s1.tool = "run_shell"; s1.effect = Effect::Irreversible;
    Step s2; s2.tool = "list_dir";  s2.effect = Effect::Pure;
    mid.push_back(s1); mid.push_back(s2);
    check(first_misplaced_irreversible(mid) == 0,
          "a non-final irreversible step is located (so the plan is refused)");

    std::vector<Step> last;
    Step t1; t1.tool = "list_dir";  t1.effect = Effect::Pure;
    Step t2; t2.tool = "run_shell"; t2.effect = Effect::Irreversible;
    last.push_back(t1); last.push_back(t2);
    check(first_misplaced_irreversible(last) == std::string::npos,
          "an irreversible FINAL step is allowed");
  }

  // ---- R2 ----
  std::printf("-- R2: a failing later step rolls back the earlier ones --\n");
  {
    std::vector<Ran> ran;
    Step m; m.tool = "mkdir"; m.args = jpath(base + "/made");
    Ran r1; const std::string o1 = run_one(m, &r1); ran.push_back(r1);
    std::printf("  step 1 mkdir: %s\n", o1.c_str());
    check(fs::exists(base + "/made"), "mkdir created the directory");
    check(r1.undo_path == base + "/made", "the new directory is marked for undo");

    Step c; c.tool = "copy";
    c.args = jcopy2(base + "/nope.txt", base + "/copy.txt");
    Ran r2; const std::string o2 = run_one(c, &r2); ran.push_back(r2);
    std::printf("  step 2 copy:  %s\n", o2.c_str());
    check(!r2.ok, "step 2 failed as intended");

    check(rollback(ran) == 0, "rollback reported no stuck steps");
    check(!fs::exists(base + "/made"), "the directory created by step 1 was removed");
  }

  // ---- R2b ----
  std::printf("-- R2b: rollback must not remove what it did not create --\n");
  {
    fs::create_directories(base + "/pre", ec);
    std::vector<Ran> ran;
    Step m; m.tool = "mkdir"; m.args = jpath(base + "/pre");
    Ran r1; const std::string o = run_one(m, &r1); ran.push_back(r1);
    std::printf("  mkdir on an existing dir: %s\n", o.c_str());
    check(r1.undo_path.empty(), "a pre-existing directory is NOT marked for undo");
    rollback(ran);
    check(fs::exists(base + "/pre"), "the pre-existing directory survived the rollback");

    // The trap that made this pass by luck. tool_mkdir answers an existing
    // directory with "already exists: <path>", which does NOT start with "error:".
    // A prefix test therefore read it as a SUCCESS. Assert the two facts
    // separately so a future edit to tool_mkdir's wording cannot quietly restore
    // the bug: (a) the output is not an error, yet (b) nothing was created.
    check(o.rfind("error:", 0) != 0, "mkdir on an existing dir is not an error");
    check(o.rfind("already exists", 0) == 0, "mkdir on an existing dir says so");
    check(!r1.ok || r1.undo_path.empty(),
          "a non-error result that created nothing is never marked for undo");
  }

  // ---- R2c ----
  std::printf("-- R2c: copy rollback --\n");
  {
    std::vector<Ran> ran;
    Step c; c.tool = "copy";
    c.args = jcopy2(base + "/src/a.txt", base + "/made.txt");
    Ran r1; const std::string o = run_one(c, &r1); ran.push_back(r1);
    std::printf("  copy: %s\n", o.c_str());
    check(r1.ok, "copy succeeded");
    check(fs::exists(base + "/made.txt"), "the destination exists");
    check(rollback(ran) == 0, "rollback reported no stuck steps");
    check(!fs::exists(base + "/made.txt"), "the destination this plan created was removed");
  }

  // ---- dependency analysis: $N references ----
  std::printf("-- dependency: a later step uses an earlier step's output --\n");
  {
    // A directory whose files have deliberately different sizes, so "largest"
    // has exactly one right answer and it is not also the first or last entry.
    const std::string d = base + "/sized";
    fs::create_directories(d, ec);
    const char* files[][2] = {{"a_small.txt", "x"}, {"b_large.txt", "0123456789ABCDEF"}, {"c_mid.txt", "0123"}};
    for (auto& f : files) {
      std::FILE* fh = std::fopen((d + "/" + f[0]).c_str(), "wb");
      std::fputs(f[1], fh);
      std::fclose(fh);
    }

    std::vector<std::string> outs;
    std::string v, why;

    // Step 1 output, produced by the real tool.
    const std::string listing = tool_list_dir(jpath(d));
    outs.push_back(listing);
    std::printf("  step 1 (list_dir): %s\n", listing.substr(0, 40).c_str());

    check(select_entry(listing, "largest", &v, &why), "largest is selectable");
    check(v == "b_large.txt", "largest picked the 16-byte file, not the first entry");
    why.clear();
    check(select_entry(listing, "first", &v, &why) && v == "a_small.txt",
          "first picked the alphabetically first entry");
    why.clear();
    check(select_entry(listing, "smallest", &v, &why) && v == "a_small.txt",
          "smallest picked the 1-byte file");

    // Substitution into a real argument string.
    //
    // The first version substituted into jpath(...) -- a ONE-argument template --
    // and then called tool_copy on it, which of course reported missing "src". The
    // test was wrong, not the substitution: an argument template must be built for
    // the tool that will consume it.
    std::string sub, werr;
    const std::string tmpl = jcopy2(d + "/$1:largest", base + "/picked.txt");
    check(substitute_refs(tmpl, outs, &sub, &werr), "$1:largest substituted");
    check(sub.find("b_large.txt") != std::string::npos,
          "the substituted argument names the largest file");

    // And the substituted argument actually drives the tool.
    const std::string copied = tool_copy(jcopy2(d + "/$1:largest", base + "/picked2.txt"));
    std::printf("  copy with a literal $1 (not substituted): %s\n", copied.c_str());
    check(copied.rfind("error:", 0) == 0,
          "an UNSUBSTITUTED reference is rejected by the tool, not silently accepted");

    const std::string copied2 = tool_copy(sub);
    std::printf("  copy with the reference substituted:     %s\n", copied2.c_str());
    check(copied2.rfind("copied", 0) == 0, "the substituted call succeeds");
    check(fs::exists(base + "/picked.txt"), "the copy exists");
    check(fs::file_size(base + "/picked.txt", ec) == 16, "it copied the 16-byte file");
  }

  std::printf("-- separator table self-check --\n");
  {
    // Whether the tie-break in split_plan is live is a property of the TABLE, not
    // of the loop. If no separator is a prefix of another, the tie-break can never
    // fire, and a mutation aimed at it would pass -- which is how this was found.
    //
    // The check does not demand a prefix pair; it REPORTS one, so that a future
    // addition of " and " alongside " and then " is visible immediately.
    static const char* kS[] = {
      " and after that ", " and afterwards ", " after which ", " and finally ",
      " and then ", " and next ", ", then ", " ; ", "; ", " then ",
      " \xe7\x84\xb6\xe5\x90\x8e", " \xe6\x8e\xa5\xe7\x9d\x80",
    };
    const int n = (int)(sizeof(kS) / sizeof(kS[0]));
    int prefixes = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        const std::string a = kS[i], b = kS[j];
        if (b.size() > a.size() && b.compare(0, a.size(), a) == 0) {
          ++prefixes;
          std::printf("  note: %s is a prefix of %s\n", kS[i], kS[j]);
        }
      }
    }
    std::printf("  -> %d prefix pair(s); tie-break is %s\n\n",
                prefixes, prefixes ? "LIVE" : "unreachable today");
    // No assertion on the count: either answer is legitimate. What is NOT
    // legitimate is not knowing which one holds.
  }
  std::printf("-- wording coverage: which separators does split_plan recognise --\n");
  {
    struct W { const char* text; int want; const char* why; };
    static const W kW[] = {
      // The one separator the implementation was written around.
      {"list C:/a and then read C:/b",                       2, "and then"},
      // Everything below is phrasing a person would plausibly use.
      {"list C:/a, then read C:/b",                          2, ", then"},
      {"list C:/a then read C:/b",                           2, "then, no and"},
      {"list C:/a and after that read C:/b",                 2, "and after that"},
      {"list C:/a and afterwards read C:/b",                 2, "and afterwards"},
      {"list C:/a and next read C:/b",                       2, "and next"},
      {"list C:/a after which read C:/b",                    2, "after which"},
      {"list C:/a and finally read C:/b",                    2, "and finally"},
      {"first list C:/a, then read C:/b",                    2, "first ... , then"},
      {"list C:/a and read C:/b",                            2, "bare and"},
      {"list C:/a; read C:/b",                               2, "semicolon"},
      {"\xe5\x88\x97\xe5\x87\xba C:/a \xe7\x84\xb6\xe5\x90\x8e\xe8\xaf\xbb\xe5\x8f\x96 C:/b", 2, "然后"},
      {"\xe5\x88\x97\xe5\x87\xba C:/a \xe6\x8e\xa5\xe7\x9d\x80\xe8\xaf\xbb\xe5\x8f\x96 C:/b", 2, "接着"},
      // A single request must NOT be split, however it is punctuated.
      {"read C:/a",                                          1, "one request"},
      {"go ahead and run hostname",                          1, "politeness and"},
    };
    int recognised = 0, total = 0;
    for (const W& w : kW) {
      const auto parts = split_plan(w.text);
      const int got = (int)parts.size();
      ++total;
      if (got == w.want) ++recognised;
      std::printf("  %-6s %-40s -> %d part(s), want %d  (%s)\n",
                  got == w.want ? "[ ok ]" : "[MISS]", w.text, got, w.want, w.why);
    }
    std::printf("  -> %d/%d phrasing(s) split as intended\n\n", recognised, total);
    // NOT a threshold: this number is the measurement. It exists so that the
    // gap between what users say and what split_plan accepts is visible before
    // the feature is wired into the dialogue path, rather than after.
  }
  std::printf("-- ambiguous bare-and: two actions, one clause -> REFUSE --\n");
  {
    // The rule that makes leaving " and " out of split_plan safe. A request that
    // detect_multi_action calls two actions, but that split_plan renders as one
    // clause, is AMBIGUOUS: the separator was there, the phrase table just could not
    // name it. Executing the single clause would act on half the request.
    //
    // So the plan is refused, and the refusal says why. This is the same asymmetry
    // as everywhere else in this file: a rephrase costs the user a turn, acting on
    // half a request costs correctness.
    struct A { const char* text; bool want_refuse; const char* why; };
    static const A kA[] = {
      {"list C:/a and read C:/b",                  true,  "two actions, one clause"},
      {"read C:/a and write C:/b",                 true,  "two actions, one clause"},
      {"go ahead and run hostname",                false, "politeness filler, one action"},
      {"list C:/a and then read C:/b",             false, "explicit separator, two clauses"},
      {"read C:/a",                                false, "one action"},
    };
    for (const A& c : kA) {
      const bool two = detect_multi_action(c.text);
      const bool one_clause = (split_plan(c.text).size() == 1);
      const bool refuse = two && one_clause;
      check(refuse == c.want_refuse, c.why);
      if (refuse != c.want_refuse) std::printf("    %s\n", c.text);
    }
  }
  std::printf("-- execute_plan: the whole path, not a hand-built vector --\n");
  {
    const std::string b2 = base + "/exec";
    fs::remove_all(b2, ec);
    fs::create_directories(b2 + "/src", ec);
    { std::FILE* f = std::fopen((b2 + "/src/a.txt").c_str(), "wb"); std::fputs("hello", f); std::fclose(f); }
    // (f) needs its own sized directory. Forgetting it made the case fail for a
    // reason that had nothing to do with the code under test -- the evidence was
    // "the earlier step produced no entries", which is the right message for the
    // wrong cause.
    fs::create_directories(b2 + "/sized", ec);
    { std::FILE* f = std::fopen((b2 + "/sized/small.txt").c_str(), "wb"); std::fputs("x", f); std::fclose(f); }
    { std::FILE* f = std::fopen((b2 + "/sized/big.txt").c_str(), "wb"); std::fputs("0123456789ABCDEF", f); std::fclose(f); }

    {
      std::vector<Step> p;
      Step s1; s1.tool = "mkdir"; s1.effect = Effect::Compensable; s1.args = jpath(b2 + "/d1");
      Step s2; s2.tool = "mkdir"; s2.effect = Effect::Compensable; s2.args = jpath(b2 + "/d2");
      Step s3; s3.tool = "copy";  s3.effect = Effect::Compensable;
      s3.args = jcopy2(b2 + "/src/a.txt", b2 + "/d1/copied.txt");
      p.push_back(s1); p.push_back(s2); p.push_back(s3);
      const PlanResult r = execute_plan(p);
      std::printf("  (a) three steps: outcome=%d stuck=%d\n", (int)r.outcome, r.stuck);
      check(r.outcome == PlanOutcome::Ok, "a three-step plan succeeds");
      check(r.outputs.size() == 3, "all three produced output");
      check(fs::exists(b2 + "/d1/copied.txt"), "the third step really copied into d1");
    }

    {
      std::vector<Step> p;
      Step s1; s1.tool = "mkdir"; s1.effect = Effect::Compensable; s1.args = jpath(b2 + "/keep1");
      Step s2; s2.tool = "copy";  s2.effect = Effect::Compensable;
      s2.args = jcopy2(b2 + "/missing.txt", b2 + "/x.txt");
      Step s3; s3.tool = "mkdir"; s3.effect = Effect::Compensable; s3.args = jpath(b2 + "/never");
      p.push_back(s1); p.push_back(s2); p.push_back(s3);
      const PlanResult r = execute_plan(p);
      std::printf("  (b) fail at 2: outcome=%d failed_at=%d stuck=%d\n",
                  (int)r.outcome, r.failed_at, r.stuck);
      check(r.outcome == PlanOutcome::StepFailed, "the plan reports a failed step");
      check(r.failed_at == 1, "it reports which step failed");
      check(r.stuck == 0, "the rollback was clean");
      check(!fs::exists(b2 + "/keep1"), "step 1 was undone");
      check(!fs::exists(b2 + "/never"), "step 3 never ran");
    }

    {
      std::vector<Step> p;
      Step s1; s1.tool = "mkdir"; s1.effect = Effect::Compensable; s1.args = jpath(b2 + "/r1");
      Step s2; s2.tool = "mkdir"; s2.effect = Effect::Compensable; s2.args = jpath(b2 + "/r2");
      Step s3; s3.tool = "copy";  s3.effect = Effect::Compensable;
      s3.args = jcopy2(b2 + "/missing.txt", b2 + "/y.txt");
      p.push_back(s1); p.push_back(s2); p.push_back(s3);
      const PlanResult r = execute_plan(p);
      std::printf("  (c) fail at 3: failed_at=%d stuck=%d\n", r.failed_at, r.stuck);
      check(r.failed_at == 2, "the third step is the one that failed");
      check(r.stuck == 0, "rollback clean");
      check(!fs::exists(b2 + "/r1"), "step 1 undone");
      check(!fs::exists(b2 + "/r2"), "step 2 undone");
    }

    {
      std::vector<Step> p;
      Step s1; s1.tool = "mkdir"; s1.effect = Effect::Compensable; s1.args = jpath(b2 + "/ref1");
      Step s2; s2.tool = "copy";  s2.effect = Effect::Compensable;
      s2.args = jcopy2(b2 + "/$1:largest", b2 + "/z.txt");
      p.push_back(s1); p.push_back(s2);
      const PlanResult r = execute_plan(p);
      std::printf("  (d) bad reference: outcome=%d why=%s\n", (int)r.outcome, r.why.c_str());
      check(r.outcome == PlanOutcome::BadReference, "a bad reference is a distinct outcome");
      check(r.failed_at == 1, "it points at the step with the reference");
      check(!fs::exists(b2 + "/ref1"), "the earlier step was rolled back too");
    }

    {
      std::vector<Step> p;
      Step s1; s1.tool = "run_shell"; s1.effect = Effect::Irreversible; s1.args = "{}";
      Step s2; s2.tool = "mkdir";     s2.effect = Effect::Compensable; s2.args = jpath(b2 + "/r1x");
      p.push_back(s1); p.push_back(s2);
      const PlanResult r = execute_plan(p);
      check(r.outcome == PlanOutcome::MisplacedIrreversible,
            "execute_plan refuses a misplaced irreversible step");
      check(r.failed_at == 0, "and names it");
      check(!fs::exists(b2 + "/r1x"), "nothing ran, so nothing exists");
    }

    {
      std::vector<Step> p;
      Step s1; s1.tool = "list_dir"; s1.effect = Effect::Pure; s1.args = jpath(b2 + "/sized");
      Step s2; s2.tool = "copy";     s2.effect = Effect::Compensable;
      s2.args = jcopy2(b2 + "/sized/$1:largest", b2 + "/winner.txt");
      p.push_back(s1); p.push_back(s2);
      const PlanResult r = execute_plan(p);
      std::printf("  (f) list-then-copy-biggest: outcome=%d\n", (int)r.outcome);
      check(r.outcome == PlanOutcome::Ok, "the two-step dependent plan succeeds");
      check(fs::exists(b2 + "/winner.txt"), "something was copied");
    }

    fs::remove_all(b2, ec);
  }
  std::printf("-- pre-flight gate: one uncertain step refuses the whole plan --\n");
  {
    const std::string b3 = base + "/gate";
    fs::remove_all(b3, ec);
    fs::create_directories(b3, ec);

    // A synthetic gate, so this tests the ORDERING property without needing a
    // memory file: any clause containing the word "unclear" is unconfirmed.
    auto conf_gate = [](const std::string& clause, std::string* why) -> bool {
      if (clause.find("unclear") != std::string::npos) {
        *why = "the clause contains a word this gate treats as unconfirmed";
        return false;
      }
      return true;
    };

    // (g) gate says yes for all -> plan runs.
    {
      std::vector<Step> p;
      Step s1; s1.tool = "mkdir"; s1.effect = Effect::Compensable;
      s1.args = jpath(b3 + "/gd1"); s1.prompt = "make a directory";
      Step s2; s2.tool = "mkdir"; s2.effect = Effect::Compensable;
      s2.args = jpath(b3 + "/gd2"); s2.prompt = "make another directory";
      p.push_back(s1); p.push_back(s2);
      const PlanResult r = execute_plan(p, +conf_gate);
      std::printf("  (g) all confirmed: outcome=%d\n", (int)r.outcome);
      check(r.outcome == PlanOutcome::Ok, "a fully confirmed plan runs");
      check(fs::exists(b3 + "/gd1") && fs::exists(b3 + "/gd2"), "both steps took effect");
    }

    // (h) THE POINT: step 2 is unconfirmed -> NOTHING runs, not even step 1.
    {
      std::vector<Step> p;
      Step s1; s1.tool = "mkdir"; s1.effect = Effect::Compensable;
      s1.args = jpath(b3 + "/never1"); s1.prompt = "make a directory";
      Step s2; s2.tool = "mkdir"; s2.effect = Effect::Compensable;
      s2.args = jpath(b3 + "/never2"); s2.prompt = "this is unclear";
      p.push_back(s1); p.push_back(s2);
      const PlanResult r = execute_plan(p, +conf_gate);
      std::printf("  (h) step 2 unconfirmed: outcome=%d failed_at=%d\n",
                  (int)r.outcome, r.failed_at);
      check(r.outcome == PlanOutcome::Unconfirmed, "the plan is refused as unconfirmed");
      check(r.failed_at == 1, "it names the unconfirmed step");
      check(!fs::exists(b3 + "/never1"),
            "STEP 1 DID NOT RUN -- the gate is pre-flight, not per-step");
      check(!fs::exists(b3 + "/never2"), "step 2 did not run either");
    }

    // (i) no gate -> the old behaviour, unchanged.
    {
      std::vector<Step> p;
      Step s1; s1.tool = "mkdir"; s1.effect = Effect::Compensable;
      s1.args = jpath(b3 + "/ng"); s1.prompt = "anything unclear at all";
      p.push_back(s1);
      const PlanResult r = execute_plan(p);
      check(r.outcome == PlanOutcome::Ok, "a null gate means assume-confident");
      check(fs::exists(b3 + "/ng"), "and the step ran");
    }

    fs::remove_all(b3, ec);
  }
  std::printf("-- the REAL gate: memory-backed confidence per clause --\n");
  {
    // The synthetic gate above proves the ORDERING. This proves the gate has a
    // real implementation: each clause is run through the same retrieval the
    // single-step path uses, and a clause whose best match is below the floor is
    // treated as unconfirmed.
    static const std::string amp = "build/ckpt/canticle.am";
    if (!fs::exists(amp)) {
      std::printf("  SKIP: no memory file at %s (run am-build first)\n", amp.c_str());
    } else {
      // WHAT THIS CAN AND CANNOT ASSERT.
      //
      // am_query_public returns the retrieval decision only: rc==0 means "a
      // candidate cleared the floor", NOT "the request would be acted on". The
      // payload-level refusals (-6 for the wrong kind of payload, the command
      // vocabulary test) live in am_main.cpp's query_reranked and are deliberately
      // NOT applied here.
      //
      // Measured: "what is the best practice in TCP/IP" returns rc=0 from
      // am_query_public (sim=0.323) while am-chat correctly declines to act on it.
      // The first version of this test asserted that am_query_public would refuse,
      // failed, and was WRONG rather than the code being wrong -- the gate is about
      // retrieval confidence, and that is all it can be asked about here.
      //
      // So the assertions are:
      //   - in-scope clauses clear their floor (the gate would let them through)
      //   - an out-of-scope clause scores LOWER than any in-scope one, which is the
      //     signal a caller would use -- not that it is refused here.
      float best_oos = -1.f, worst_in = 2.f;
      struct Probe { const char* clause; bool in_scope; };
      static const Probe kP[] = {
        {"list the directory C:/tmp",            true},
        {"read the file C:/tmp/a.txt",           true},
        {"what is the best practice in TCP/IP",  false},
        {"tell me a joke about programmers",     false},
      };
      for (const Probe& p : kP) {
        std::string reply; float sim = 0;
        const int rc = am_query_public(amp, p.clause, reply, sim);
        std::printf("  rc=%-2d sim=%.3f  in_scope=%d  %s\n", rc, sim, (int)p.in_scope, p.clause);
        if (p.in_scope) {
          if (sim < worst_in) worst_in = sim;
          check(rc == 0, "an in-scope clause clears the retrieval floor");
        } else {
          if (sim > best_oos) best_oos = sim;
        }
      }
      std::printf("  worst in-scope %.3f  vs  best out-of-scope %.3f\n", worst_in, best_oos);
      check(worst_in > best_oos,
            "every in-scope clause scores above every out-of-scope one");
    }
  }
  std::printf("-- end to end: sentence -> clauses -> steps -> execution --\n");
  {
    static const std::string amp2 = "build/ckpt/canticle.am";
    if (!fs::exists(amp2)) {
      std::printf("  SKIP: no memory file\n");
    } else {
      // ABSOLUTE paths, and the reason is a real limitation rather than tidiness.
      //
      // token_is_path accepts a relative path only when its LAST segment has an
      // extension (or it starts with ./ or ../). "build/_plan/e2e/sized" therefore
      // fails the file-reference gate and the clause is refused with -6:
      //
      //     list the directory build/_plan/e2e/sized   -> refused -6
      //     list the directory ./build/x               -> list_dir
      //     list the directory C:/.../sized            -> list_dir
      //
      // That rule is deliberate: a bare "a/b" is far more often prose (TCP/IP,
      // C++/Rust) than a directory, and the licence exists because "TCP/IP" was once
      // read as a path. Widening it here to make this test tidier would trade a
      // measured false-positive rate for convenience in a test.
      //
      // Recorded because it is a REAL limitation for multi-step plans: "list the
      // directory build/logs and then ..." does not work unless the user writes
      // "./build/logs". See ROADMAP stage 3.
      char absbuf[512];
      std::string b4 = _fullpath(absbuf, (base + "/e2e").c_str(), sizeof absbuf)
                           ? std::string(absbuf) : (base + "/e2e");
      // FORWARD SLASHES ONLY. _fullpath returns backslashes, and those then travel
      // into a JSON argument string where each one is an escape character: the tool
      // received "C:Userscccanticleuild_plane2e/sized" and reported a syntax error
      // naming a path nobody wrote.
      //
      // Normalising here rather than in jpath() because this is where the OS hands
      // back a Windows path; every other path in this file is already forward-slash.
      for (char& ch : b4) if (ch == (char)92) ch = '/';
      fs::remove_all(b4, ec);
      fs::create_directories(b4 + "/sized", ec);
      { std::FILE* f = std::fopen((b4 + "/sized/small.txt").c_str(), "wb"); std::fputs("x", f); std::fclose(f); }
      { std::FILE* f = std::fopen((b4 + "/sized/big.txt").c_str(), "wb"); std::fputs("0123456789ABCDEF", f); std::fclose(f); }

      {
        const NamedStep n = name_clause(amp2, "list the directory C:/tmp");
        std::printf("  name_clause(list the directory): ok=%d tool=%s sim=%.3f\n",
                    (int)n.ok, n.step.tool.c_str(), n.sim);
        check(n.ok, "a clear clause resolves to a tool");
        check(n.step.tool == "list_dir", "and to the RIGHT tool");
        check(n.step.effect == Effect::Pure, "and carries the effect from the table");
      }
      {
        const NamedStep n = name_clause(amp2, "tell me a joke about programmers");
        std::printf("  name_clause(joke): ok=%d why=%s\n", (int)n.ok, n.why.c_str());
        check(!n.ok || n.sim < 0.5f,
              "an out-of-scope clause does not resolve confidently");
      }

      {
        const std::string request =
            std::string("list the directory ") + b4 + "/sized and then copy " + b4 + "/sized/big.txt to " + b4 + "/picked.txt";
        const auto clauses = split_plan(request);
        std::printf("  split -> %d clause(s)\n", (int)clauses.size());
        check(clauses.size() == 2, "the request split into two");

        std::vector<Step> steps;
        for (const auto& c : clauses) {
          const NamedStep n = name_clause(amp2, c);
          std::printf("    [%s] %s\n", n.ok ? "ok " : "MISS", c.c_str());
          if (!n.ok) std::printf("           why=%s  tool=[%s]\n", n.why.c_str(), n.step.tool.c_str());
          if (n.ok) steps.push_back(n.step);
        }
        check(steps.size() == clauses.size(), "every clause named a step");

        if (steps.size() == clauses.size()) {
          const PlanResult r = execute_plan(steps);
          std::printf("  execute: outcome=%d\n", (int)r.outcome);
          check(r.outcome == PlanOutcome::Ok, "the whole chain succeeds");
          check(fs::exists(b4 + "/picked.txt"), "the file was actually copied");
        }
      }

      fs::remove_all(b4, ec);
    }
  }
  // ---- the failure directions, which are the ones that matter ----
  std::printf("-- dependency: every unresolvable reference FAILS --\n");
  {
    std::vector<std::string> outs;
    outs.push_back("");                                  // step 1 produced nothing
    std::string v, why;
    check(!select_entry("", "largest", &v, &why), "no entries -> largest cannot resolve");
    check(!why.empty(), "and the reason is not empty");

    check(!substitute_refs("$2", outs, &v, &why), "a forward reference fails");
    std::printf("    forward ref: %s\n", why.c_str());

    check(!substitute_refs("$0", outs, &v, &why), "step 0 does not exist");
    check(!substitute_refs("$1:largest", outs, &v, &why),
          "referring to an empty earlier output fails");

    std::vector<std::string> failed;
    failed.push_back("error: cannot find C:/nope");
    check(!substitute_refs("$1:first", failed, &v, &why),
          "referring to a FAILED step fails");
    std::printf("    failed step: %s\n", why.c_str());

    std::vector<std::string> sized;
    sized.push_back("[F] a.txt  (no size reported)\n");
    check(!select_entry(sized[0], "largest", &v, &why),
          "largest fails when no entry reported a size");
    std::printf("    no sizes:   %s\n", why.c_str());

    std::vector<std::string> good;
    good.push_back("[F] a.txt  (1 B)\n");
    check(!substitute_refs("$1:bogus", good, &v, &why), "an unknown selector fails");
    std::printf("    bad selector: %s\n", why.c_str());

    // A bare dollar sign is not a reference and must pass through.
    std::string out;
    check(substitute_refs("cost is $ and $x", good, &out, &why), "a bare $ is not a reference");
    check(out == "cost is $ and $x", "and it is copied through unchanged");
  }
  // ---- splitting ----
  std::printf("-- plans are split and named --\n");
  {
    const std::string p = "list the directory C:/tmp and then read the file C:/tmp/a.txt";
    const auto parts = split_plan(p);
    check(parts.size() == 2, "the phrase split into two clauses");
    if (parts.size() == 2) {
      check(parts[0].find("list") != std::string::npos, "clause 1 kept its verb");
      check(parts[1].find("read") != std::string::npos, "clause 2 kept its verb");
      Step s;
      check(plan_step(parts[0], &s) && s.tool == "list_dir", "clause 1 names list_dir");
      check(plan_step(parts[1], &s) && s.tool == "read_file", "clause 2 names read_file");
    }
  }

  fs::remove_all(base, ec);

  std::printf("\n  %s  (%d checks, %d failure(s))\n",
              g_fails ? "FAILED" : "ALL PASS", g_checks, g_fails);
  return g_fails ? 1 : 0;
}

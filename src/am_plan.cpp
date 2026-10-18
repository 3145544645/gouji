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

std::vector<std::string> split_plan(const std::string& text) {
  const std::string sep = " and then ";
  std::vector<std::string> out;
  size_t at = 0;
  while (true) {
    const size_t next = text.find(sep, at);
    if (next == std::string::npos) { out.push_back(text.substr(at)); break; }
    out.push_back(text.substr(at, next - at));
    at = next + sep.size();
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

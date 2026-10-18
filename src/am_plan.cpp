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
//  WHAT THIS DOES NOT DO: no dependency analysis. Step 2 does not receive step
//  1's output, so 'list C:/log then copy the biggest one' cannot work. Plans
//  are sequences of INDEPENDENT operations -- strictly smaller than the ROADMAP
//  example, and the measurement says so rather than implying otherwise.
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

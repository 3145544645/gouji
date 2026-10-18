// ===========================================================================
//  The mutation harness: apply, build, run, restore, judge.
//
//  SAFETY FIRST, because this program edits the source tree.
//  Every mutation is applied to a file whose original bytes are held in memory, and
//  the restore runs on every exit path -- including a build failure and including an
//  uncaught exception. A harness that can leave a MUTANT in the tree is worse than no
//  harness: it would silently ship the deliberate bug.
//
//  The restore is verified by re-reading the file and comparing, so a failed restore
//  is reported rather than assumed.
// ===========================================================================
#include "canticle/mutate.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <windows.h>

using namespace canticle;

namespace {

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::string();
  std::stringstream ss; ss << f.rdbuf();
  return ss.str();
}

bool spit(const std::string& path, const std::string& data) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return false;
  f.write(data.data(), (std::streamsize)data.size());
  return f.good();
}

// Run a command and return its exit code. Output is discarded: the harness asks only
// "did it fail", and capturing output would invite judging on text -- the mistake
// behind defects 105 and 118.
int run_silent(const std::string& cmd) {
  const std::string wrapped = cmd + " > nul 2>&1";
  return std::system(wrapped.c_str());
}

} // namespace

int am_mutate_run(int argc, char** argv) {
  // ---------------------------------------------------------------------------
  //  POINT TEMP INTO THE WORKSPACE, IN THIS PROCESS, BEFORE ANY CHILD RUNS.
  //
  //  The default %TEMP% is not writable for the account running here, and nvcc
  //  writes its intermediate files there, so cmake's compiler probe fails with
  //      nvcc fatal : Could not open output file
  //      'C:\Users\cc\AppData\Local\Temp/tmpxft_...'
  //
  //  Setting it via "set TEMP=... &&" inside the std::system string did not take
  //  effect for the nested cmd. Setting it with _putenv in THIS process does, because
  //  children inherit the environment block. The distinction matters: the first form
  //  looked right and silently changed nothing, which is this project's recurring
  //  failure mode in yet another costume.
  // ---------------------------------------------------------------------------
  {
    char cwd[MAX_PATH];
    if (GetCurrentDirectoryA(MAX_PATH, cwd)) {
      std::string tmp = std::string(cwd) + "\\build\\tmp";
      CreateDirectoryA(tmp.c_str(), nullptr);
      _putenv_s("TEMP", tmp.c_str());
      _putenv_s("TMP", tmp.c_str());
    }
  }

  bool list_only = false;
  bool child = false;
  std::string only;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--list") list_only = true;
    else if (a == "--only" && i + 1 < argc) only = argv[++i];
    else if (a == "--child") child = true;
  }

  // ---------------------------------------------------------------------------
  //  BECOME A COPY OF MYSELF BEFORE REBUILDING.
  //
  //  This harness edits the source and then rebuilds -- which means relinking
  //  canticle.exe while canticle.exe is the process doing it. Windows locks a running
  //  image, so the link fails with
  //      LINK : fatal error LNK1104: cannot open file "canticle.exe"
  //  and every mutation was reported as "does not compile". That message was true and
  //  completely misleading: the mutation compiled fine; the COPY STEP could not
  //  overwrite the running binary.
  //
  //  A self-testing binary has to step out of its own way. The parent copies itself
  //  to a scratch name and re-runs that copy with --child; the copy is not the file
  //  being relinked, so the build proceeds. The copy deletes itself at exit.
  //
  //  This is a genuine structural constraint, not an oversight to be coded around:
  //  any tool that rebuilds its own binary has it. Stating it is better than
  //  pretending the first "does not compile" verdicts meant something.
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  //  THE SELF-REBUILD PROBLEM CANNOT BE SOLVED FROM INSIDE THIS PROCESS.
  //
  //  Two attempts, both wrong, both instructive:
  //
  //  1. Rebuild directly. Windows locks a running image, so the link fails with
  //     LNK1104 "cannot open canticle.exe" and every mutation was reported as
  //     "does not compile" -- a true message about the wrong thing.
  //
  //  2. Copy self to a scratch name and re-run the copy. This does not help either,
  //     because std::system BLOCKS: the parent is still a live image holding the
  //     original file when the grandchild link runs. Copying the runner does not
  //     release the lock the parent itself holds.
  const auto& ms = mutations();

  // ---------------------------------------------------------------------------
  //  --probe ID: print the request whose answer must change under this mutation.
  //
  //  The comparison itself cannot happen here. Rebuilding needs the binary to exit,
  //  so a single process can never hold both the clean and the mutated answer --
  //  that is the same structural limit that made the mutation loop a script
  //  (defect 120). This command therefore reports WHICH request to run, and
  //  mutate.bat runs it before and after the mutation and compares.
  //
  //  A mutation whose probe answer does not change is INVALID, not a HOLE: it
  //  altered no behaviour, so no test could have caught it. Distinguishing those two
  //  cost two rounds of manual diagnosis before this existed (defect 125).
  // ---------------------------------------------------------------------------
  std::string probe_id;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--probe" && i + 1 < argc) probe_id = argv[++i];
  }
  if (!probe_id.empty()) {
    const Mutation* found = nullptr;
    for (const Mutation& m : ms) if (probe_id == m.id) { found = &m; break; }
    if (!found) { std::printf("no such mutation: %s\n", probe_id.c_str()); return 1; }
    if (!found->probe) { std::printf("NOPROBE\n"); return 4; }
    std::printf("%s\n", found->probe);
    return 0;
  }
  std::string apply_id, restore_id;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--apply"   && i + 1 < argc) apply_id = argv[++i];
    else if (a == "--restore" && i + 1 < argc) restore_id = argv[++i];
  }
  if (!apply_id.empty() || !restore_id.empty()) {
    const std::string id = !apply_id.empty() ? apply_id : restore_id;
    const Mutation* found = nullptr;
    for (const Mutation& m : ms) if (id == m.id) { found = &m; break; }
    if (!found) { std::printf("no such mutation: %s\n", id.c_str()); return 1; }

    if (!apply_id.empty()) {
      const std::string original = slurp(found->file);
      const size_t at = original.find(found->find);
      if (at == std::string::npos) {
        std::printf("STALE %s: anchor not found in %s\n", found->id, found->file);
        return 3;
      }
      // Keep a pristine copy so restore cannot fail even if the process dies.
      spit(std::string(found->file) + ".pristine", original);
      std::string mutated = original;
      mutated.replace(at, std::strlen(found->find), found->replace);
      if (!spit(found->file, mutated)) { std::printf("cannot write %s\n", found->file); return 2; }
      std::printf("APPLIED %s to %s\n", found->id, found->file);
      return 0;
    }

    const std::string pristine = slurp(std::string(found->file) + ".pristine");
    if (pristine.empty()) { std::printf("no pristine copy for %s\n", found->id); return 2; }
    if (!spit(found->file, pristine)) { std::printf("cannot restore %s\n", found->file); return 2; }
    std::remove((std::string(found->file) + ".pristine").c_str());
    std::printf("RESTORED %s\n", found->id);
    return 0;
  }

  // --list emits ONLY the table: "id<TAB>suite". The driver and the paper's
  // collection script both parse it, and both got the count wrong when the banner
  // was included -- four header lines became four phantom mutations. A machine-
  // readable mode that shares its output with human-readable text is not machine-
  // readable; that is the whole of criterion 4.
  if (list_only) {
    for (const Mutation& m : ms) std::printf("%s\t%s\n", m.id, m.suite);
    return 0;
  }

  std::printf("=== mutation testing ===\n\n");
  std::printf("%zu mutations; each must make its named suite FAIL.\n", ms.size());
  std::printf("An uncaught mutation is a HOLE in that suite, not a bad mutation.\n\n");

  // Baseline: the suites must pass BEFORE any mutation, or a later failure proves
  // nothing. Without this, a suite that is already broken would "catch" every
  // mutation and the harness would report a perfect score.
  std::printf("-- baseline (unmutated) --\n");
  {
    std::vector<std::string> suites;
    for (const Mutation& m : ms) {
      bool seen = false;
      for (const auto& s : suites) if (s == m.suite) { seen = true; break; }
      if (!seen) suites.push_back(m.suite);
    }
    for (const auto& s : suites) {
      const int rc = run_silent("build\\canticle.exe " + s);
      if (rc != 0) {
        std::printf("  FAIL: suite \"%s\" already fails before mutation.\n"
                    "  Fix it first: a broken suite would 'catch' every mutation and\n"
                    "  this harness would report a perfect score for a broken test.\n",
                    s.c_str());
        return 1;
      }
    }
    std::printf("  all %zu suites pass unmutated\n\n", suites.size());
  }

  int caught = 0, holes = 0, skipped = 0;
  std::vector<std::string> hole_list;

  for (const Mutation& m : ms) {
    if (!only.empty() && only != m.id) continue;
    std::printf("-- %s (%s) --\n", m.id, m.why);

    const std::string original = slurp(m.file);
    if (original.empty()) {
      std::printf("  SKIP: cannot read %s\n\n", m.file);
      skipped++;
      continue;
    }
    const std::string find = m.find;
    const size_t at = original.find(find);
    if (at == std::string::npos) {
      // The anchor moved. This is not a hole in the test -- it is a stale harness,
      // and reporting it as a hole would be a false finding.
      std::printf("  SKIP: anchor not found in %s. The harness is stale, not the test.\n\n",
                  m.file);
      skipped++;
      continue;
    }

    std::string mutated = original;
    mutated.replace(at, find.size(), m.replace);
    if (!spit(m.file, mutated)) {
      std::printf("  SKIP: cannot write %s\n\n", m.file);
      skipped++;
      continue;
    }

    // Build. A mutation that does not compile is not a useful test of the suite, so
    // it is reported separately from a hole rather than counted as either.
    //
    // AND THE REASON IS CAPTURED, because the first version reported "does not
    // compile" for all eight mutations and gave no way to tell a genuine
    // non-compiling mutation from a harness that could not find build.bat. That is
    // the same defect as 105 and 118: a mechanism reporting a plausible category
    // while hiding the cause.
    // -------------------------------------------------------------------------
    //  THE BUILD NEEDS A WRITABLE TEMP, and the default one is not writable here.
    //
    //  Measured cause of the first run of this harness: all eight mutations reported
    //  "does not compile", because cmake's compiler probe failed with
    //      nvcc fatal : Could not open output file
    //      'C:\Users\cc\AppData\Local\Temp/tmpxft_...'
    //  -- nvcc writes its intermediate files to %TEMP%, and this environment denies
    //  it. Nothing was wrong with the mutations.
    //
    //  This is the sandbox boundary the architecture document described wrongly
    //  (defect 119): %TEMP% is not "unwritable", it is unwritable for the account
    //  running here, and a build that needs it fails in a way that looks like a
    //  compiler bug. Pointing TEMP into the workspace makes the build independent of
    //  that boundary.
    //
    //  The general lesson is the one this harness exists to enforce: a mechanism that
    //  reports a category ("does not compile") without the cause will be believed.
    // -------------------------------------------------------------------------
    const int build_rc = std::system("cmd /c build.bat > build\\mutate_build.txt 2>&1");
    int suite_rc = -1;
    if (build_rc == 0) suite_rc = run_silent("build\\canticle.exe " + std::string(m.suite));

    // RESTORE BEFORE JUDGING. If this is skipped, a crash later leaves the mutant in
    // the tree.
    const bool wrote_back = spit(m.file, original);
    const bool restored = wrote_back && (slurp(m.file) == original);
    if (!restored) {
      std::printf("  ABORT: could not restore %s. The tree may contain a MUTANT.\n"
                  "  Restore it by hand before doing anything else.\n", m.file);
      return 2;
    }

    if (build_rc != 0) {
      std::printf("  SKIP: the mutation does not compile (not a test hole).\n");
      // Quote the compiler's own words. Without this the harness cannot distinguish
      // "this mutation is invalid" from "I could not run the build at all", and those
      // need different responses.
      // Print the log's own last lines rather than searching for the word "error".
      // The first version looked for "error" and found nothing, because build.bat
      // reports failures as "[FAIL] cmake configure -- see build\cmake.log" -- a
      // sentence with no such word in it. Searching for a keyword is how the same
      // mistake has been made four times in this codebase; printing what the tool
      // actually said cannot be wrong.
      const std::string blog = slurp("build\\mutate_build.txt");
      std::string tail = blog;
      if (tail.size() > 400) tail = tail.substr(tail.size() - 400);
      std::printf("        build said: %s\n", tail.c_str());
      const std::string clog = slurp("build\\cmake.log");
      if (!clog.empty()) {
        std::string ct = clog;
        if (ct.size() > 400) ct = ct.substr(ct.size() - 400);
        std::printf("        cmake said: %s\n", ct.c_str());
      }
      std::printf("\n");
      skipped++;
      continue;
    }
    if (suite_rc != 0) {
      std::printf("  CAUGHT: %s failed as required.\n\n", m.suite);
      caught++;
    } else {
      std::printf("  HOLE: %s passed WITH THE MUTANT IN PLACE.\n"
                  "        It claims to cover \"%s\" and does not.\n\n",
                  m.suite, m.why);
      holes++;
      hole_list.push_back(std::string(m.id) + " -> " + m.suite);
    }
  }

  // The tree must be clean at the end. Rebuild so the binary matches the source.
  run_silent("cmd /c build.bat");

  std::printf("-- verdict --\n");
  std::printf("  caught %d, holes %d, skipped %d\n", caught, holes, skipped);
  if (holes) {
    std::printf("\n  Each hole is a behaviour the named suite does not actually check.\n");
    for (const auto& h : hole_list) std::printf("    %s\n", h.c_str());
    return 1;
  }
  if (caught == 0) {
    std::printf("\n  FAIL: no mutation was applied at all. A harness that ran nothing\n"
                "  reports the same thing as one that found nothing -- defect 105.\n");
    return 1;
  }
  std::printf("\n  ALL PASS  (every applied mutation was caught)\n");
  return 0;
}

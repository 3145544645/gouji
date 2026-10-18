// ===========================================================================
//  am-atomic: a failed tool must not leave a partial result behind.
//
//  WHY THIS IS ITS OWN SUITE
//  ------------------------
//  Every other suite asserts on STRINGS -- which tool was chosen, what the
//  payload was. This one asserts on the FILESYSTEM: after a failed copy, what
//  exists? Those are different failure modes and they need different tests. A
//  string test cannot see a half-populated directory.
//
//  THE DEFECT THIS WAS WRITTEN FOR
//  ------------------------------
//  copy of a directory that fails partway left the already-copied entries at the
//  destination. Measured: src/{a,b,c}.txt with c.txt held open by another process
//  gave
//      error: cannot copy directory: ... being used by another process
//  and dst/a.txt, dst/b.txt were left behind.
//
//  Two things are wrong with that beyond the leftover itself:
//    1. the caller was told it failed, so the leftover is a surprise;
//    2. a retry then failed with "destination already exists", so the failure
//       was not recoverable by repeating the request.
//
//  ROADMAP stage 3 names "a mid-step failure leaves dirty state" as the hard part
//  of multi-step planning. It was already true of a SINGLE-step command, which is
//  the reason to fix it before building anything that chains steps.
//
//  HOW THE FAILURE IS PRODUCED
//  ---------------------------
//  A file is opened with a share mode that denies read/write to other handles.
//  That is a real, reproducible mid-copy failure on Windows -- not an injected
//  error flag. If the copy somehow succeeds anyway the test says so rather than
//  passing vacuously.
// ===========================================================================
#include "canticle/model.h"   // tool_copy and the other tool entry points
#include <cstdio>
#include <string>
#include <filesystem>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;
using namespace canticle;

namespace {

int g_total = 0, g_fails = 0;
void check(bool ok, const char* what) {
  ++g_total;
  if (!ok) { ++g_fails; std::printf("  FAIL %s\n", what); }
}

// A directory with three files, one of which can be locked against readers.
void make_tree(const std::string& base) {
  std::error_code ec;
  fs::remove_all(base, ec);
  fs::create_directories(base + "/src", ec);
  for (const char* n : {"a.txt", "b.txt", "c.txt"}) {
    std::ofstream f(base + "/src/" + n, std::ios::binary);
    f << n;
  }
}

std::string copy_args(const std::string& src, const std::string& dst) {
  return std::string("{\"src\":\"") + src + "\",\"dst\":\"" + dst + "\"}";
}

} // namespace

int am_atomic_run(int, char**) {
  std::printf("=== a failed tool leaves nothing behind ===\n\n");
  const std::string base = "build/_atomic";

  // ---- case 1: directory copy fails partway, destination must not appear ----
  std::printf("-- directory copy, one file locked --\n");
  make_tree(base);
  const std::string dst = base + "/dst";
  bool locked = false;
#ifdef _WIN32
  // Deny both read and write sharing: any attempt to open c.txt for reading fails.
  HANDLE h = CreateFileA((base + "/src/c.txt").c_str(), GENERIC_READ, 0, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  locked = (h != INVALID_HANDLE_VALUE);
#endif
  // IF THIS FAILS, THE TEST IS INVALID, not the code. A failed lock means the
  // setup did not create the condition the case is about, so the copy below would
  // succeed and every assertion after it would pass vacuously. Measured once in
  // CI-like sequential runs: a previous process still held the handle.
  //
  // Reported as its own named failure rather than folded into the others, because
  // "the lock was not acquired" and "the copy left a partial tree" are opposite
  // diagnoses and must not look the same.
  check(locked, "SETUP: the lock was acquired (if this fails the case proves nothing, retry)");

  const std::string r = tool_copy(copy_args(base + "/src", dst));
  const bool failed = (r.rfind("error:", 0) == 0);
  std::printf("  result: %s\n", r.c_str());
  check(failed, "the copy reported failure");
  check(!fs::exists(dst), "no partial destination was left behind");
  check(!fs::exists(dst + ".canticle-partial"), "no staging directory was left behind");

#ifdef _WIN32
  if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
#endif

  // ---- case 2: the retry works. This is the point of the rollback. ----
  std::printf("-- retry after the failure --\n");
  const std::string r2 = tool_copy(copy_args(base + "/src", dst));
  const bool ok2 = (r2.rfind("copied", 0) == 0);
  std::printf("  result: %s\n", r2.c_str());
  check(ok2, "retrying after a rolled-back failure succeeds");
  check(fs::exists(dst + "/a.txt") && fs::exists(dst + "/c.txt"),
        "the successful copy has all three files");

  // ---- case 3: success is unchanged (the fix must not break the happy path) ----
  std::printf("-- single file copy still works --\n");
  const std::string one = base + "/one.txt";
  const std::string r3 = tool_copy(copy_args(base + "/src/a.txt", one));
  std::printf("  result: %s\n", r3.c_str());
  check(r3.rfind("copied", 0) == 0, "single-file copy succeeds");
  check(fs::exists(one), "the file exists");
  check(!fs::exists(one + ".canticle-partial"), "no staging file was left behind");

  // ---- case 4: refusing to overwrite is still a refusal, not a half-write ----
  std::printf("-- destination exists --\n");
  const std::string r4 = tool_copy(copy_args(base + "/src/a.txt", one));
  std::printf("  result: %s\n", r4.c_str());
  check(r4.find("already exists") != std::string::npos, "an existing destination is refused");
  check(!fs::exists(one + ".canticle-partial"), "the refusal left no staging file");

  std::error_code ec;
  fs::remove_all(base, ec);

  std::printf("\n  %s  (%d checks, %d failure(s))\n",
              g_fails ? "FAILED" : "ALL PASS", g_total, g_fails);
  return g_fails ? 1 : 0;
}

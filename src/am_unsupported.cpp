// ===========================================================================
//  Operations the system understands but cannot perform.
//
//  WHY THIS IS SEPARATE FROM "unsupported input"
//  ---------------------------------------------
//  Two different things produce a refusal, and telling them apart is the point:
//
//    UNKNOWN       the request is not recognised. Nothing in the structure layer
//                  claims it and no stored pattern is close. The honest answer is
//                  "I do not understand this".
//
//    UNSUPPORTED   the request IS understood. "delete C:/tmp/x.txt" names one
//                  operation unambiguously. The system simply has no tool for it,
//                  and the honest answer is "I understand, and I do not do that".
//
//  Before the distinction existed, these requests were absorbed by whatever rule
//  happened to match. Measured:
//
//      "delete C:/tmp/scratch.txt"    -> read_file   <- destructive request, read back
//      "rename C:/tmp/a.txt to ..."   -> read_file
//      "make a directory C:/tmp/newdir" -> list_dir
//      "how big is C:/.../STATUS.md"  -> list_dir
//      "find all txt files in C:/..." -> list_dir   <- the FILTER is silently dropped
//
//  The first is the dangerous one: a user asked to remove a file and the system
//  answered by opening it. Nothing was harmed, but only because read_file is
//  harmless -- the routing itself carried no understanding of the verb.
// ===========================================================================
#include "canticle/am.h"
#include <cstdio>
#include <string>

using namespace canticle;

static int fails = 0, total = 0;

static void expect(Intent got, Intent want, const char* what) {
  total++;
  if (got != want) {
    std::printf("  FAIL %-46s got %-12s want %s\n", what, intent_name(got),
                intent_name(want));
    fails++;
  }
}

int am_unsupported_run(int, char**) {
  std::printf("=== unsupported operations ===\n\n");

  std::printf("-- must classify as Unsupported, not something else --\n");
  expect(infer_intent("delete C:/tmp/scratch.txt"),          Intent::Unsupported, "delete file");
  expect(infer_intent("remove C:/tmp/a.txt"),                Intent::Unsupported, "remove file");
  expect(infer_intent("rename C:/tmp/a.txt to C:/tmp/b.txt"),Intent::Unsupported, "rename");
  // MOVED OUT, fourth time: copy is now implemented. The transition is asserted in
  // its new home below so that "was refused, now served" is re-checked every run
  // rather than only being true once.
  std::printf("-- supported requests must be unaffected --\n");
  expect(infer_intent("read the file C:/tmp/a.txt"),         Intent::ReadFile,  "read");
  expect(infer_intent("list the directory C:/tmp"),          Intent::ListDir,   "list");
  expect(infer_intent("write hello to C:/tmp/a.txt"),        Intent::WriteFile, "write");
  expect(infer_intent("run the command hostname"),           Intent::RunShell,  "shell");
  expect(infer_intent("what is 40 - 14"),                    Intent::Calc,      "calc");

  std::printf("-- capabilities that moved from this list into real tools --\n");
  // A capability leaving this list is a NORMAL event: it happens every time a tool
  // is implemented. What must not happen is the old assertion staying behind, which
  // reports correct behaviour as a failure. These four lines exist so that the
  // current boundary is stated in one place and any future move is a one-line edit.
  expect(infer_intent("how big is C:/tmp/a.txt"),            Intent::Stat,   "stat");
  expect(infer_intent("find all txt files in C:/tmp"),       Intent::Search, "search");
  expect(infer_intent("make a directory C:/tmp/nd"),         Intent::MkDir,  "mkdir");
  expect(infer_intent("copy C:/tmp/a.txt to C:/tmp/b.txt"),  Intent::Copy,   "copy");
  expect(infer_intent("search for TODO in C:/tmp/src"),      Intent::Search, "grep/search");
  expect(infer_intent("make a directory C:/tmp/newdir"),     Intent::MkDir, "make a directory");
  expect(infer_intent("create a folder C:/tmp/x"),           Intent::MkDir, "create a folder");
  expect(infer_intent("mkdir C:/tmp/x"),                     Intent::MkDir, "mkdir verb");
  expect(infer_intent("new folder C:/tmp/x"),                Intent::MkDir, "new folder");

  std::printf("-- the trap: 'remove' vs 'remove from' --\n");
  // "read the file" must not be captured by the destructive table just because a
  // later word resembles one. These are controls for the word-boundary rule.
  expect(infer_intent("read the file C:/tmp/notes.txt"),     Intent::ReadFile,  "read, safe");
  expect(infer_intent("list the directory C:/tmp/removed"),  Intent::ListDir,   "path containing 'removed'");

  std::printf("\n  %s  (%d failure(s))\n", fails ? "FAILED" : "ALL PASS", fails);
  return fails ? 1 : 0;
}

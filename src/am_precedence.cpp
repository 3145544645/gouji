// ===========================================================================
//  Rule precedence.
//
//  Nineteen defects were ordering mistakes, all with the same cause: which rule
//  won was decided by the ORDER OF LINES in infer_intent, so a rule added later
//  silently lost to an earlier one that matched something incidental.
//
//  The precedence table is written out in am.cpp. This file makes it EXECUTABLE:
//  each case below is a pair of signals that could both fire, and the assertion is
//  which one must win. A future edit that reorders the rules breaks one of these
//  rather than quietly changing routing.
//
//  The general principle being tested:
//      a rule describing an OPERATION outranks a rule observing a SYMBOL.
// ===========================================================================
#include "canticle/am.h"
#include "canticle/context.h"
#include <cstdio>
#include <string>

using namespace canticle;

static int fails = 0, total = 0;

static void expect(Intent got, Intent want, const char* why) {
  total++;
  if (got != want) {
    std::printf("  FAIL %-56s got %-12s want %s\n", why, intent_name(got),
                intent_name(want));
    fails++;
  }
}

int am_precedence_run(int, char**) {
  std::printf("=== rule precedence ===\n\n");

  std::printf("-- 1. refusing outranks everything --\n");
  // "delete X" contains a path, so the bare-path default and the read verbs would
  // both fire. Neither may.
  expect(infer_intent("delete C:/tmp/a.txt"),        Intent::Unsupported, "delete beats bare path");
  expect(infer_intent("rename C:/a.txt to C:/b.txt"),Intent::Unsupported, "rename beats paths");
  // even when a read verb is also present
  expect(infer_intent("read then delete C:/tmp/a.txt"), Intent::Unsupported,
         "unsupported beats a read verb");

  std::printf("-- 2. an explicit tool outranks the generic rules --\n");
  expect(infer_intent("how big is C:/tmp/a.txt"),    Intent::Stat,   "stat beats bare path");
  expect(infer_intent("find all x in C:/tmp"),       Intent::Search, "search beats containment");
  expect(infer_intent("make a directory C:/tmp/x"),  Intent::MkDir,  "mkdir beats bare path");
  expect(infer_intent("copy C:/a.txt to C:/b.txt"),  Intent::Copy,   "copy beats bare path");

  std::printf("-- 3. containment outranks symbols and weak verbs --\n");
  // a slash plus a weak verb: the containment reading must win
  expect(infer_intent("show me what is in C:/tmp"),  Intent::ListDir, "in beats 'show'");
  expect(infer_intent("display what is in C:/tmp"),  Intent::ListDir, "in beats 'display'");
  expect(infer_intent("what is in C:/tmp"),          Intent::ListDir, "in beats bare path");
  // but containment must NOT capture a plain read
  expect(infer_intent("read the file C:/tmp"),       Intent::ReadFile, "no ' in ', so read");
  // and must not capture a shell command either
  expect(infer_intent("run the command hostname"),   Intent::RunShell, "no path, so shell");

  std::printf("-- 4. arithmetic outranks the bare-path rule --\n");
  // "3/4 of 18" contains a slash, which the path rule would claim
  expect(infer_intent("what is 3/4 of 18"),          Intent::Calc, "arithmetic beats slash-as-path");
  expect(infer_intent("what is 40 - 14"),            Intent::Calc, "plain arithmetic");

  std::printf("-- the two boundaries that actually broke --\n");
  //
  // An independent audit pointed out that this suite covered levels 1,2,3,5,6,7 but
  // omitted the two conflicts that had actually produced bugs. A suite that asserts
  // only the cases its author already understood protects the wrong half.
  //
  //   defect 65:    strong verb vs containment -- "put hello in C:/tmp/x" became a
  //                 directory listing on the first attempt at the table.
  //   defect 99:    arithmetic vs containment -- "what is 7 + 8 in C:/tmp" produced
  //                 list_dir {"path":"7+8"}.
  expect(infer_intent("put hello in C:/tmp/todo.json"), Intent::WriteFile,
         "strong verb beats containment (defect 65)");
  {
    const Intent got = infer_intent("what is 7 + 8 in C:/tmp");
    total++;
    if (got == Intent::ListDir) {
      std::printf("  FAIL %-56s got list_dir; arithmetic may not become a listing\n",
                  "arithmetic vs containment");
      fails++;
    }
  }

  std::printf("-- 5/6/7. verbs, weak hints, and the bare-path default --\n");
  expect(infer_intent("read the file C:/tmp/a.txt"), Intent::ReadFile, "strong verb");
  expect(infer_intent("show me C:/tmp/a.txt"),       Intent::ReadFile, "weak hint");

  // A bare path returns UNKNOWN, and that is deliberate rather than a gap.
  //
  // The bare-path default (level 7) applies to a path with no other signal --
  // "inspect D:/x.md" -- and the ROUTER then treats it as a read. infer_intent
  // itself reports Unknown for a path-only turn so that the session layer can
  // inherit the previous turn's tool instead of forcing a read: "and C:/work/b.txt"
  // following a directory listing must stay a listing.
  //
  // The distinction matters for reading this test: Unknown here does NOT mean the
  // request is unanswered. It means routing is deferred to context, and with no
  // context the router's fallback is read.
  expect(infer_intent("C:/tmp/a.txt"),               Intent::Unknown, "bare path defers to context");
  expect(infer_intent("inspect D:/repo/zeta.md"),    Intent::ReadFile, "bare path with a request verb reads");

  std::printf("-- the context licences, directly --\n");
  //
  // The licence is the mechanism the whole section rests on, so it is asserted on its
  // own rather than only through the router. Both directions matter: a licence that
  // rejected everything would pass the negative cases and break every path request.
  {
    struct Case { const char* text; bool want; const char* why; };
    static const Case kCases[] = {
      {"read the file C:/tmp/a.txt",         true,  "drive prefix"},
      {"read the file D:/data/alpha.md",     true,  "drive prefix"},
      {"read the file build/ckpt/x.am",      true,  "extension on the last segment"},
      {"read the file STATUS.md",            true,  "bare filename with extension"},
      {"list the directory /tmp",            true,  "leading separator"},
      {"read the file ./x.txt",              true,  "explicit relative prefix"},
      {"compare the tradeoffs in C++/Rust",  false, "C++ is not a volume name"},
      {"the best practice in TCP/IP",        false, "no drive, no extension"},
      {"the difference in A/B testing",      false, "no drive, no extension"},
      {"what is 7 + 8",                      false, "arithmetic, not a path"},
    };
    for (const Case& c : kCases) {
      const bool got = context::licensed_as_path(c.text);
      total++;
      if (got != c.want) {
        std::printf("  FAIL %-56s got %s, want %s (%s)\n", c.text,
                    got ? "path" : "not-path", c.want ? "path" : "not-path", c.why);
        fails++;
      }
    }
  }

  std::printf("\n  %s  (%d failure(s))\n", fails ? "FAILED" : "ALL PASS", fails);
  return fails ? 1 : 0;
}

// ===========================================================================
//  Multi-action request detection.
//
//  WHY A DETECTOR AND NOT AN ANSWER
//  --------------------------------
//  The corpus has zero multi-action examples, so asking for two things produced a
//  MIXED call: the tool from the second clause with the payload from the first.
//
//      "read the file X and list the directory Y"   ->  list_dir {"path":"X"}
//      "what is 12 * 7 then list the directory Y"   ->  list_dir {"path":"12*7"}
//      "first read X then run the command hostname" ->  read_file {"path":"hostname"}
//
//  Valid JSON, clean execution, wrong action. Teaching the model to emit two calls
//  would need data that does not exist and a protocol that runs one call per turn;
//  recognising the pattern needs neither, and refusing is recoverable.
//
//  THE HARD PART IS NOT FINDING CONJUNCTIONS
//  -----------------------------------------
//  "read A and B" and "read A and list B" both contain " and ". The discriminator
//  is whether the second clause is itself a REQUEST. Two tests are needed:
//    * a verb in the second clause ("and list Y");
//    * a bare command name in the second clause ("hostname and whoami"), where
//      neither half has a verb because the commands ARE the clauses.
//  And a false positive is costly -- it would reject a legitimate single request
//  whose FILENAME contains the word "and".
// ===========================================================================
#include "canticle/multiact.h"
#include <cstdio>
#include <string>

using namespace canticle;

static int fails = 0, total = 0;

static void expect(bool got, bool want, const char* what) {
  total++;
  if (got != want) {
    std::printf("  FAIL %-52s got %s want %s\n", what, got ? "MULTI" : "single",
                want ? "MULTI" : "single");
    fails++;
  }
}

int am_multiact_run(int, char**) {
  std::printf("=== multi-action detection ===\n\n");

  std::printf("-- must be detected as two actions --\n");
  expect(detect_multi_action("read the file C:/a.txt and list the directory C:/tmp"), true,
         "read ... and list ...");
  expect(detect_multi_action("what is 12 * 7 then list the directory C:/tmp"), true,
         "calc then list");
  expect(detect_multi_action("first read C:/a.txt then run the command hostname"), true,
         "first ... then run ...");
  expect(detect_multi_action("hostname and whoami"), true,
         "two bare command names");
  expect(detect_multi_action("read the file C:/a.txt and also show me C:/b.txt"), true,
         "and also show");
  expect(detect_multi_action("list C:/tmp then what is 3 + 4"), true,
         "list then calc");

  std::printf("-- must NOT be flagged --\n");
  expect(detect_multi_action("read the file C:/Users/cc/canticle/STATUS.md"), false,
         "single read");
  expect(detect_multi_action("list the directory C:/tmp"), false,
         "single list");
  expect(detect_multi_action("run the command hostname"), false,
         "single shell");
  expect(detect_multi_action("what is 40 - 14"), false,
         "single expression");
  // The trap: "and" inside a filename or a path must not split the request.
  expect(detect_multi_action("read the file D:/data/alpha_and_beta.txt"), false,
         "filename containing 'and'");
  expect(detect_multi_action("list C:/work/black_and_white"), false,
         "directory name containing 'and'");
  expect(detect_multi_action("write hello to D:/a_and_b.log"), false,
         "write target containing 'and'");

  std::printf("\n  %s  (%d failure(s))\n", fails ? "FAILED" : "ALL PASS", fails);
  return fails ? 1 : 0;
}

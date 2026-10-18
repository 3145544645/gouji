
// ===========================================================================
//  Unit tests for the pure host-side pieces of CANTICLE-AM.
//  These exist because a one-character offset in payload extraction silently
//  produced WRONG ARITHMETIC ("7 + 8" answered "77 + 8 = 85"). A test asserts
//  it; an eyeball did not catch it.
// ===========================================================================
#include "canticle/am.h"
#include <cstdio>
#include <cstring>
#include <string>

// The shared lead-in matcher is defined in am.cpp; declared here so the unit test
// can exercise it directly instead of only through its callers.
namespace canticle {
size_t match_lead_in(const std::string& q, const char* const* phrases, size_t count);
}
#include <string>
#include <vector>

using namespace canticle;

static int fails = 0, total = 0;

static void check(const std::string& got, const std::string& want, const char* what) {
  total++;
  if (got != want) {
    std::printf("  FAIL %-40s got '%s'  want '%s'\n", what, got.c_str(), want.c_str());
    fails++;
  }
}

int amtest_run() {
  std::printf("=== CANTICLE-AM unit tests ===\n\n");

  std::printf("-- extract_payload --\n");
  check(extract_payload("what is 7 + 8"),        "7 + 8",        "simple sum");
  check(extract_payload("what is 123 * 456"),    "123 * 456",    "3-digit product");
  check(extract_payload("what is 40 - 14"),      "40 - 14",      "subtraction");
  check(extract_payload("what is 999 / 3"),      "999 / 3",      "division");
  check(extract_payload("what is 7+8"),          "7+8",          "no spaces");
  check(extract_payload("read the file C:/work/gamma.txt"),
        "C:/work/gamma.txt",                                      "windows-ish path");
  check(extract_payload("read the file /etc/hosts"), "/etc/hosts", "unix path");
  check(extract_payload("list the directory C:/tmp"), "C:/tmp",   "directory");
  check(extract_payload("run the command hostname"), "hostname",  "bare command");
  check(extract_payload("write hello to D:/data/cfg.log"),
        "D:/data/cfg.log",                                        "path with ext");
  check(extract_payload("hello"),                "hello",        "single word");

  // -------------------------------------------------------------------------
  //  Arithmetic as real people write it.
  //
  //  These forms come from CLINC150 (23703 genuine utterances). The synthetic
  //  corpus contains only "40 + 14" style input, so before this the system
  //  handled 1 of 8 real calculator requests:
  //      "what is 7 x 7"                -> list_dir {"path":"7"}
  //      "what is the sum of 10 plus 5" -> calc {"expr":"5"}
  //      "please add 456 and 781 for me"-> calc {"expr":"me"}
  //      "what is 30% off 279"          -> calc {"expr":"279"}
  // -------------------------------------------------------------------------
  // NOTE: extract_payload returns the expression as it appears, spaces included;
  // normalise_payload is what strips them, and it runs later in the pipeline.
  // The first version of these assertions expected "7*7" and reported six
  // failures against behaviour that was correct. Assert what the function
  // returns, and add one case that pins the normalisation separately.
  check(extract_payload("what is 7 x 7"),                       "7 * 7",      "x as multiply");
  check(extract_payload("what is 8 times 9"),                   "8 * 9",      "times");
  check(extract_payload("what is the sum of 10 plus 5"),        "10 + 5",     "plus");
  check(extract_payload("please add 456 and 781 for me"),       "456 + 781",  "add ... and");
  check(extract_payload("what is 20 minus 7"),                  "20 - 7",     "minus");
  check(extract_payload("calculate 144 divided by 12"),         "144 / 12",   "divided by");
  check(normalise_payload(extract_payload("what is 7 x 7")),    "7*7",        "normalised for the wire");

  // ---- the shared lead-in matcher: EARLIEST wins, longest breaks ties ----
  //
  // Three defects came from call sites doing this by hand with rfind:
  //     #16  "put " inside "output"
  //     #56  " to " inside " into "
  //     #59  "run " inside "echo a run b"
  // The fix is a single primitive, so it is asserted directly here rather than
  // only through the functions that use it.
  {
    static const char* kP[] = {"run ", "run the command "};
    // earliest position wins: "run " occurs at 0 and again at 12; the matcher must
    // return the one at 0, extended by its length, not the later one.
    const size_t a = match_lead_in("run the command netstat -an", kP, 2);
    check(std::to_string(a == std::string::npos ? -1 : (long long)a),
          std::to_string((long long)std::strlen("run the command ")),
          "longer phrase wins at the same position");
    const size_t b2 = match_lead_in("run echo a run b", kP, 2);
    check(std::to_string(b2 == std::string::npos ? -1 : (long long)b2), "4",
          "earliest occurrence wins, not the last");
    static const char* kNone[] = {"zzz "};
    check(match_lead_in("run the command x", kNone, 1) == std::string::npos ? "none" : "found",
          "none", "no match returns npos");
  }

  // ---- arithmetic written as English ----
  //
  // Measured need from CLINC150: real people write "multiply 45 by 23", not "45*23".
  // Every synthetic prompt uses symbols, so this was invisible to the whole suite.
  // Asserted through normalise_payload, because that is what the ROUTER applies
  // before comparing or emitting. extract_payload alone leaves the spaces the
  // substitution produced ("45 * 23"), while the pipeline normalises to "45*23" --
  // and asserting the un-normalised form would encode a value the system never
  // actually uses. Measured end-to-end: "can you multiply 45 by 23" emits
  //     <|tool|>calc {"expr":"45*23"}<|/tool|>
  check(normalise_payload(extract_payload("can you multiply 45 by 23")), "45*23", "multiply a by b");
  check(normalise_payload(extract_payload("can you divide 100 by 20")),  "100/20", "divide a by b");
  // order matters: "subtract 36 from 5" is 5-36, not 36-5
  check(normalise_payload(extract_payload("can you subtract 36 from 5")), "5-36", "subtract a from b");
  check(normalise_payload(extract_payload("can you subract 36 from 5")), "5-36", "misspelled 'subract'");
  check(normalise_payload(extract_payload("what is seventeen plus three")), "17+3", "number words + plus");
  check(normalise_payload(extract_payload("what is 45 times 23")), "45*23", "N times N");
  check(normalise_payload(extract_payload("what is forty five plus six")), "45+6", "compound number words");
  check(normalise_payload(extract_payload("can you add 456 and 781")), "456+781", "add a and b");

  // ---- lead-in matching inside a payload (substring hazards) ----
  //
  // The lead-in scan uses rfind, i.e. the LAST occurrence, on the assumption that
  // a lead-in phrase appears before the command. When a command's own text CONTAINS
  // such a phrase, that assumption breaks and the payload is cut in the middle.
  //
  // These cases probe the boundary rather than assuming it is safe. Where one
  // fails it is recorded as a known limitation in the test itself instead of being
  // silently accepted, so the risk stays visible.
  check(extract_payload("run the command netstat -an"),  "netstat -an", "plain command");
  check(extract_payload("execute git run status"),       "git run status", "command containing 'run'");
  check(extract_payload("run echo a run b"),             "echo a run b",  "repeated 'run' inside command");

  // ---- copy: TWO paths, both taken from the query ----
  {
    std::string s, d;
    bool ok;
    ok = extract_copy_paths("copy C:/a.txt to C:/b.txt", &s, &d);
    check(ok ? (s + "|" + d) : "FAIL", "C:/a.txt|C:/b.txt", "copy X to Y");
    ok = extract_copy_paths("copy the file C:/a.txt to C:/b.txt", &s, &d);
    check(ok ? (s + "|" + d) : "FAIL", "C:/a.txt|C:/b.txt", "copy the file X to Y");
    ok = extract_copy_paths("please copy C:/a.txt into C:/backup", &s, &d);
    check(ok ? (s + "|" + d) : "FAIL", "C:/a.txt|C:/backup", "copy X into Y");
    ok = extract_copy_paths("duplicate C:/a.txt as C:/b.txt", &s, &d);
    check(ok ? (s + "|" + d) : "FAIL", "C:/a.txt|C:/b.txt", "duplicate X as Y");
    ok = extract_copy_paths("copy D:/x/y.log to C:/tmp/z.log", &s, &d);
    check(ok ? (s + "|" + d) : "FAIL", "D:/x/y.log|C:/tmp/z.log", "both full paths");
    // Corpus-shaped input, which is what retrieval actually produces.
    ok = extract_copy_paths("kindly copy C:/log/gamma.json to C:/work/delta.txt", &s, &d);
    check(ok ? (s + "|" + d) : "FAIL", "C:/log/gamma.json|C:/work/delta.txt", "prefix + both paths");
    ok = extract_copy_paths("please copy D:/data/log.log to C:/tmp/delta.txt", &s, &d);
    check(ok ? (s + "|" + d) : "FAIL", "D:/data/log.log|C:/tmp/delta.txt", "please + both paths");
    // a request with no destination must NOT invent one
    ok = extract_copy_paths("copy C:/a.txt", &s, &d);
    check(ok ? "INVENTED" : "FAIL", "FAIL", "no destination -> refuse");
    // ---------------------------------------------------------------------------
    //  AND THE CASE THAT ACTUALLY REACHES THE FINAL RETURN.
    //
    //  Mutation testing found this the hard way: replacing the last line with
    //  "return !src->empty();" changed no verdict, because "copy C:/a.txt" has no
    //  separator at all and returns early at the "no connective" test. The mutation
    //  was aimed at code the input never reached -- defect 122's shape again.
    //
    //  A sentence WITH a connective but WITHOUT a path on the right does reach it:
    //  the split succeeds, the source is found, and the destination is not.
    // ---------------------------------------------------------------------------
    ok = extract_copy_paths("copy C:/a.txt to", &s, &d);
    check(ok ? "INVENTED" : "FAIL", "FAIL", "connective but no destination -> refuse");
    ok = extract_copy_paths("copy C:/a.txt into the folder", &s, &d);
    check(ok ? "INVENTED" : "FAIL", "FAIL", "connective, prose on the right -> refuse");

    // ---------------------------------------------------------------------------
    //  EVERY SEPARATOR IN THE TABLE NEEDS A CASE, which mutation testing showed.
    //  Removing " over to " from kSep changed no verdict, because no test used it --
    //  the suite covered four of the five separators and nothing noticed the fifth.
    //  A table with an entry no test exercises is a table entry that may as well not
    //  be there, which is the shape of defect 122 in a different place.
    // ---------------------------------------------------------------------------
    ok = extract_copy_paths("copy C:/a.txt over to C:/b.txt", &s, &d);
    check(ok ? (s + "|" + d) : "FAIL", "C:/a.txt|C:/b.txt", "copy X over to Y");
    ok = extract_copy_paths("copy C:/a.txt -> C:/b.txt", &s, &d);
    check(ok ? (s + "|" + d) : "FAIL", "C:/a.txt|C:/b.txt", "copy X -> Y");
    // 中文分隔符
    ok = extract_copy_paths("copy C:/a.txt 到 C:/b.txt", &s, &d);
    check(ok ? (s + "|" + d) : "FAIL", "C:/a.txt|C:/b.txt", "copy X 到 Y");
  }

  // ---- search: two arguments, one transferred payload ----
  check(extract_search_pattern("find all TODO in C:/src"),      "TODO",   "find all X in P");
  check(extract_search_pattern("search for main in C:/src"),    "main",   "search for X in P");
  check(extract_search_pattern("grep for config in D:/data"),   "config", "grep for X in P");
  check(extract_payload("find all TODO in C:/src"),             "C:/src", "search payload is the path");

  // Polite prefixes must not survive into the needle. The generator puts them in
  // front of every template, so any extraction that removes whole phrases has to
  // cope with them; the first version stripped the search verb only and produced
  // "can you main" and "kindly FIXME" as patterns.
  check(extract_search_pattern("can you search for main in C:/"),   "main",   "polite prefix stripped");
  check(extract_search_pattern("please search in config in C:/"),   "config", "please prefix");
  check(extract_search_pattern("hey locate main in C:/"),           "main",   "hey prefix");
  check(extract_search_pattern("kindly look for FIXME in C:/Users"),"FIXME",  "kindly prefix");
  check(extract_search_pattern("could you find all TODO in D:/x"),  "TODO",   "could you prefix");

  // ---- named operations (also verbatim from CLINC150) ----
  check(extract_payload("what is 8 factorial"),                 "8!",          "factorial");
  check(extract_payload("what is the square root of 144"),      "sqrt(144)",   "square root");
  check(extract_payload("what is the square root of 80"),       "sqrt(80)",    "square root");
  check(extract_payload("what is the square root of 1 million"),"sqrt(1*1000000)","word scale");
  check(extract_payload("what is 3/4 of 18"),                   "18*3/4",      "fraction of");
  check(extract_payload("what is 18 percent tip on 20 dollars"),"20*18/100",   "percent tip");
  check(extract_payload("what is 30% off 279"),                 "279*30/100", "percent off");
  check(extract_payload("what is 60 percent off of 350 dollars"),"350*60/100","percent word form");

  // -------------------------------------------------------------------------
  //  Multi-word shell commands.
  //
  //  The fallback rule returns the LAST whitespace token. That is right for
  //  "hostname" and wrong for anything with arguments, and the failure is silent
  //  and plausible-looking: "ipconfig /all" became {"cmd":"/all"}, which is a
  //  well-formed tool call that does the wrong thing.
  //
  //  Two rules were needed, and the second was only found by testing flags:
  //    1. a lead-in phrase ("execute ", "run the command ") marks where the
  //       command starts, and the payload is everything after it;
  //    2. that rule must be tested BEFORE the path rule, because a flag contains
  //       a slash and "cmd /c dir" would otherwise be read as the path "/c".
  // -------------------------------------------------------------------------
  check(extract_payload("run the command netstat -an"),
        "netstat -an",                            "command with a flag");
  check(extract_payload("please execute the utility ping -n 3"),
        "ping -n 3",                              "long lead-in, flag and value");
  check(extract_payload("please execute the utility ipconfig /all"),
        "ipconfig /all",                          "slash flag is not a path");
  check(extract_payload("execute cmd /c dir"),
        "cmd /c dir",                             "slash flag mid-command");
  check(extract_payload("run Get-Process | Select-Object -First 5"),
        "Get-Process | Select-Object -First 5",   "pipeline");
  // and the rules must not have stolen the path cases
  check(extract_payload("read the file C:/work/gamma.txt"),
        "C:/work/gamma.txt",                      "path still wins for read_file");

  std::printf("\n-- encoder geometry (similar text must be similar) --\n");
  {
    AmEncoder enc; enc.D = 2048;
    std::vector<std::string> docs = {
      "read the file C:/work/a.txt", "read the file C:/work/b.txt",
      "run the command hostname",    "list the directory C:/tmp",
      "what is 1 + 2",               "write hello to C:/x.log",
    };
    enc.build_idf(docs);
    auto cos = [&](const std::string& a, const std::string& b) {
      std::vector<float> x, y;
      enc.encode(a, x); enc.encode(b, y);
      double s = 0; for (size_t i = 0; i < x.size(); ++i) s += (double)x[i] * y[i];
      return s;
    };
    const double same_tmpl = cos("read the file C:/work/a.txt", "read the file C:/other/z.md");
    const double diff_tmpl = cos("read the file C:/work/a.txt", "run the command whoami");
    const double same_expr = cos("what is 12 + 34", "what is 56 + 78");
    const double diff_expr = cos("what is 12 + 34", "list the directory C:/");

    std::printf("  same template  : %.3f\n", same_tmpl);
    std::printf("  diff template  : %.3f\n", diff_tmpl);
    std::printf("  same expr shape: %.3f\n", same_expr);
    std::printf("  diff vs dir    : %.3f\n", diff_expr);
    total += 2;
    if (!(same_tmpl > diff_tmpl + 0.15)) {
      std::printf("  FAIL same-template prompts are not clearly closer\n"); fails++;
    }
    if (!(same_expr > diff_expr + 0.15)) {
      std::printf("  FAIL same-shape expressions are not clearly closer\n"); fails++;
    }

    std::printf("\n  self-similarity of identical text: %.3f (must be 1.000)\n",
                cos("read the file C:/work/a.txt", "read the file C:/work/a.txt"));
    total++;
    if (cos("read the file C:/work/a.txt", "read the file C:/work/a.txt") < 0.999) {
      std::printf("  FAIL identical text does not self-align\n"); fails++;
    }
  }

  std::printf("\n-- payload difference detection --\n");
  {
    std::vector<std::pair<std::string,std::string>> cases = {
      {"what is 7 + 8", "what is 77 + 86"},
      {"read the file C:/a.txt", "read the file C:/b.txt"},
    };
    for (auto& c : cases) {
      const std::string qp = extract_payload(c.first), mp = extract_payload(c.second);
      total++;
      if (qp == mp) {
        std::printf("  FAIL payloads wrongly equal: '%s' vs '%s'\n", qp.c_str(), mp.c_str());
        fails++;
      } else {
        std::printf("  ok  '%s' != '%s'\n", qp.c_str(), mp.c_str());
      }
    }
  }

  std::printf("\n%s  %d/%d passed\n", fails ? "[FAIL]" : "[PASS]", total - fails, total);
  return fails ? 1 : 0;
}

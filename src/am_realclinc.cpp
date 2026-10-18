// ===========================================================================
//  Real-corpus evaluation.
//
//  WHY THIS EXISTS
//  ---------------
//  am-realcase holds 26 cases and every one of them was WRITTEN BY THE SAME PERSON
//  WHO WROTE THE RULES. Measured against it, the system reads 26/26 -- but that
//  number measures agreement between two artifacts from one mind, not the ability
//  to handle a sentence nobody anticipated.
//
//  This is the same error as defect 65, where the precedence test passed because it
//  only asserted the conflicts its author happened to imagine. Writing one's own
//  exam is a specific trap, and the remedy is the same: evaluate on data that was
//  not produced here.
//
//  CLINC150 supplies 22500 genuine utterances by real people over 150 intents, in
//  the standard train/val/val_oos/test/test_oos split. Nothing in it was written
//  with this system in mind, which is exactly the point.
//
//  WHAT IS MEASURED
//  ----------------
//  Two properties, both about refusal rather than about tool choice, because tool
//  choice has no ground truth here -- CLINC150 does not know what read_file is:
//
//    1. IN-SCOPE FALSE REFUSALS. The in-scope intents that plausibly map onto a
//       capability this system has (dates, arithmetic, spelling, synonyms,
//       definitions) are collected and the system is asked to handle them. Any
//       refusal is a miss: the sentence is within reach and the system declined.
//
//    2. OUT-OF-SCOPE FALSE ACCEPTS. The oos set is 1500 requests for things this
//       system does not do (weather, banking, restaurants, travel). Any tool CALL
//       is a miss: acting on a request it cannot serve is the unrecoverable
//       direction -- a wrong answer can be corrected, a wrong action cannot.
//
//  The asymmetry is deliberate and matches the refusal design: in-scope misses are
//  recoverable, out-of-scope actions are not.
// ===========================================================================
#include "canticle/am.h"
#include "canticle/clinc.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <sstream>

using namespace canticle;

namespace {

// ---------------------------------------------------------------------------
//  WHAT "IN SCOPE" MEANS HERE, and why the list is short.
//
//  THE FIRST VERSION OF THIS LIST WAS WRONG, and the way it was wrong is worth
//  recording. It held 34 CLINC150 intents -- date, time, spelling, definition,
//  synonyms, translate, flip_coin, roll_dice, unit conversions, exchange rates --
//  on the reasoning that they "sound like things a tool could do".
//
//  They are not things THESE tools do. This system has nine tools: read, write,
//  list, shell, calc, stat, search, mkdir, copy. It is a file-and-shell router.
//  Every intent on that list except arithmetic genuinely has no tool, so refusing
//  them is CORRECT, and a test that counts those refusals as misses reports correct
//  behaviour as failure.
//
//  That is defect 53's mistake exactly, in a new place: a test expectation that
//  encodes a capability the system does not have.
//
//  So the list holds only the intents whose requests genuinely land on a tool, and
//  it is short because the intersection really is short:
//
//      calculator   arithmetic -> calc
//      date, time   a shell command can answer these, and run_shell exists
//
//  The rest of CLINC150 is out of scope for this system by construction, which is
//  itself the finding: the overlap between a general-assistant dataset and a
//  file-router is nearly empty, and the meaningful measurement on this corpus is
//  the OUT-OF-SCOPE property, not the in-scope one.
//
//  In-scope coverage of the system's OWN capabilities is measured by am-eval and
//  friends, on prompts shaped like the ones the tools actually serve.
// ---------------------------------------------------------------------------
//  THE SECOND CORRECTION TO THIS LIST, and the second time the mistake was the
//  same: listing an intent because the CAPABILITY sounds plausible rather than
//  because the SENTENCES land on a tool.
//
//  date and time were removed for exactly that reason. Measured:
//
//      "which day is it today"                  no shell verb, no path -> refused
//      "would you let me know today's date"     a question, not a command
//      "tell me the date it will be 5 days from now"
//
//  run_shell COULD answer these, but only if the user asked for a command to be run.
//  A request phrased as a question about the world is not a request to execute
//  anything, and refusing it is right. Counting those refusals as misses reported
//  correct behaviour as failure -- defect 53, for the third time, in a new place.
//
//  What remains is the one intent whose sentences really do land on a tool, and the
//  check on that is mechanical rather than intuitive: the sentence must NAME the
//  operation. "multiply 45 by 23" names it; "which day is it" does not.
//
//  THE GENERAL RULE, and it is worth stating because three separate mistakes came
//  from violating it:
//      an intent is in scope when its SENTENCES reach a tool, not when its SUBJECT
//      sounds like something a tool could handle.
struct Group { const char* clinc_intent; const char* capability; };
const Group kInScope[] = {
  {"calculator", "arithmetic -> calc"},
};

} // namespace

// ===========================================================================
//  CHINESE OUT-OF-SCOPE, same property, different language.
//
//  WHY A SECOND LANGUAGE MATTERS
//  -----------------------------
//  Every Chinese prompt in the synthetic corpus was written by the same hand that
//  wrote the Chinese keyword tables, so the two share their vocabulary by
//  construction -- the identical flaw that hid the English gap (defect 90).
//
//  corpus/alpaca_zh.json is 48818 Chinese instructions written by other people.
//  What it does NOT contain is file-and-shell requests: measured, its "compute the
//  area of a rectangle" and "list three deserts" sentences are general-assistant
//  questions, and this system is a file router. The overlap is as empty in Chinese
//  as it was in CLINC150.
//
//  That makes the corpus useless for measuring in-scope coverage and VALUABLE for
//  the out-of-scope property -- which is the one that matters, because acting on a
//  request that cannot be served is the direction that cannot be undone.
//
//  So this mode asks one question: does the system decline Chinese questions about
//  the world as reliably as it declines English ones? Any answer is informative --
//  if the rate is much worse than the English 9.0%, then the Chinese side of the
//  refusal path is weaker and nothing measured it before.
// ===========================================================================
static int run_chinese_oos(const std::string& am, const std::string& path, int sample) {
  std::printf("=== Chinese out-of-scope (alpaca_zh) ===\n\n");

  // ---------------------------------------------------------------------------
  //  Read the alpaca layout DIRECTLY rather than converting the file first.
  //
  //  parse_clinc expects [text,label] pairs; this file is an array of objects with
  //  "instruction"/"input"/"output" fields. A conversion step would work, but it
  //  would also be a second piece of code between the data and the measurement --
  //  and the last such step silently paired every sentence with the wrong label
  //  while reporting a perfectly plausible total (defect 86).
  //
  //  Scanning for the "instruction" key and taking the next string is narrow enough
  //  to be obviously correct, and it asserts its own shape below by reporting how
  //  many strings it found.
  // ---------------------------------------------------------------------------
  std::vector<std::string> prompts;
  {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      std::printf("  cannot open %s\n", path.c_str());
      return 1;
    }
    std::stringstream ss; ss << f.rdbuf();
    const std::string s = ss.str();
    static const char* kKey = "\"instruction\"";
    size_t i = 0;
    while ((i = s.find(kKey, i)) != std::string::npos) {
      size_t c = s.find(':', i + std::strlen(kKey));
      if (c == std::string::npos) break;
      size_t a = s.find('"', c);
      if (a == std::string::npos) break;
      size_t b = a + 1;
      std::string v;
      while (b < s.size() && s[b] != '"') {
        if (s[b] == '\\' && b + 1 < s.size()) { v.push_back(s[b+1]); b += 2; continue; }
        v.push_back(s[b]); ++b;
      }
      if (b >= s.size()) break;
      if (!v.empty()) prompts.push_back(v);
      i = b + 1;
    }
  }
  std::printf("parsed %zu \"instruction\" strings from %s\n\n", prompts.size(), path.c_str());
  if (prompts.empty()) {
    std::printf("  FAIL: no instructions found -- check the key name and the file.\n");
    return 1;
  }

  int tested = 0, acted = 0;
  std::vector<std::string> acted_on;
  for (size_t i = 0; i < prompts.size() && tested < sample; ++i) {
    const std::string& q = prompts[i];
    if (q.size() < 4 || q.size() > 80) continue;
    std::string reply; float sim = 0;
    if (am_query_public(am, q, reply, sim) != 0) continue;
    tested++;
    if (reply.find("<|tool|>") != std::string::npos) {
      acted++;
      if (acted_on.size() < 12) acted_on.push_back("    \"" + q + "\"");
    }
  }

  std::printf("tested %d Chinese requests, %d acted on (%.1f%%)\n",
              tested, acted, tested ? 100.0 * acted / tested : 0.0);
  for (const auto& s : acted_on) std::printf("%s\n", s.c_str());

  std::printf("\n-- verdict --\n");
  if (tested == 0) { std::printf("  FAIL: nothing tested.\n"); return 1; }
  const double rate = 100.0 * acted / tested;
  std::printf("  Chinese out-of-scope acted on : %.1f%%  (ceiling 15.0%%)\n", rate);

  // ---------------------------------------------------------------------------
  //  A ZERO COUNT IS AN INTERVAL, not a point.
  //
  //  Reporting "0.0%" from 160 samples invites the reading "this never happens".
  //  What 160 samples with no failures actually support is the upper bound given by
  //  the rule of three: with zero events in n trials, the 95% confidence interval
  //  for the true rate runs from 0 to about 3/n. At n=160 that is 1.9%.
  //
  //  The distinction matters here because the whole point of this measurement is a
  //  claim about reliability, and "0.0%" overstates what a sample this size can
  //  support. The corpus has 48818 instructions; the sample is small because each
  //  query costs a full retrieval. Stating the interval keeps the claim honest
  //  without needing more compute, and names what a larger run would buy.
  // ---------------------------------------------------------------------------
  if (acted == 0) {
    const double upper = 100.0 * 3.0 / (double)tested;
    std::printf("  zero failures in %d samples -> true rate is below about %.1f%%\n"
                "  (rule of three, 95%% confidence). The corpus holds 48818 instructions;\n"
                "  a larger --per narrows this bound.\n", tested, upper);
  }
  if (rate > 15.0) {
    std::printf("  FAIL: Chinese questions about the world are being acted on far more\n"
                "  often than English ones. The refusal path is language-dependent.\n");
    return 1;
  }
  std::printf("\n  ALL PASS\n");
  return 0;
}

int am_realclinc_run(int argc, char** argv) {
  const std::string am = "build\\ckpt\\canticle.am";
  std::string path = "corpus\\clinc150.json";
  int per_intent = 20;
  bool chinese = false;
  // ---------------------------------------------------------------------------
  //  PARSE EVERY FLAG BEFORE ACTING ON ANY.
  //
  //  The first version returned as soon as it saw --chinese, inside the parsing
  //  loop. So "--chinese --per 400" tested 160 sentences and reported a bound that
  //  400 samples would have narrowed -- the flag that controls the sample size was
  //  silently ignored, and the output still said "a larger --per narrows this
  //  bound".
  //
  //  This is the FOURTH instance of an early return swallowing later work in this
  //  codebase (defects 51, 57, 87). The remedy is the same each time: do not act
  //  from inside the loop that reads the arguments. Collect, then dispatch.
  // ---------------------------------------------------------------------------
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--path" && i + 1 < argc) path = argv[++i];
    else if (a == "--per" && i + 1 < argc) per_intent = atoi(argv[++i]);
    else if (a == "--chinese") chinese = true;
  }
  if (chinese) return run_chinese_oos(am, "corpus\\alpaca_zh.json", per_intent * 8);

  // The file is a JSON object keyed by split name. parse_clinc reads flat
  // [text,label] pairs, so it is used once per split: the scan is order-independent,
  // which means the boundary between splits does not affect which pairs are found.
  const auto items = parse_clinc(path);
  std::printf("=== real-corpus evaluation (CLINC150) ===\n\n");
  std::printf("parsed %zu [text,intent] pairs from %s\n\n", items.size(), path.c_str());
  if (items.empty()) {
    std::printf("  nothing parsed -- check the path and the file shape.\n");
    return 1;
  }

  // Bucket by intent.
  std::map<std::string, std::vector<std::string>> by_intent;
  for (const auto& it : items) by_intent[it.label].push_back(it.text);

  std::printf("distinct intents: %zu\n", by_intent.size());
  //
  // PARSER SELF-CHECK, added because the first run of this file reported a correct
  // total (23703 pairs) together with an impossible second statistic (19748 distinct
  // labels). The count was right and the data was wrong, which is the hardest kind
  // of error to notice -- so the shape is asserted here rather than trusted.
  if (by_intent.size() < 100 || by_intent.size() > 300) {
    std::printf("\n  ABORT: %zu distinct labels. CLINC150 has ~151 (150 intents + oos);\n"
                "  a much larger number means the parser is pairing keys with texts\n"
                "  rather than texts with labels, and every figure below would be\n"
                "  computed from nonsense.\n", by_intent.size());
    return 1;
  }

  // ---- property 1: in-scope requests must not be refused ----
  std::printf("\n-- in-scope intents: a refusal is a miss --\n");
  //
  // Only three intents are listed, and the reason is in the comment on kInScope:
  // CLINC150's other 147 intents ask for things this system has no tool for, so
  // refusing them is the DESIGNED behaviour, not a miss. Measuring them here would
  // report correct refusals as failures -- defect 53's error in a new place.
  //
  // The honest in-scope figure for a file-and-shell router therefore cannot come
  // from this corpus. It comes from am-eval, whose prompts are shaped like the
  // requests the tools serve. What THIS file contributes is the other direction:
  // how often a general-assistant request is acted on when it should be declined.
  int in_tested = 0, in_refused = 0;
  std::vector<std::string> in_misses;
  for (const Group& g : kInScope) {
    const auto it = by_intent.find(g.clinc_intent);
    if (it == by_intent.end()) continue;
    std::printf("  %-18s (%zu sentences, %s)\n", g.clinc_intent,
                it->second.size(), g.capability);
    for (int k = 0; k < per_intent && k < (int)it->second.size(); ++k) {
      const std::string& q = it->second[k];
      std::string reply; float sim = 0;
      if (am_query_public(am, q, reply, sim) != 0) continue;
      in_tested++;
      // "refused" is defined as: no tool call produced. That is the observable
      // fact and does not depend on the wording of any message.
      if (reply.find("<|tool|>") == std::string::npos) {
        in_refused++;
        in_misses.push_back("    \"" + q + "\"");
      }
    }
  }
  std::printf("\n  tested %d in-scope sentences, %d refused (%.1f%%)\n",
              in_tested, in_refused,
              in_tested ? 100.0 * in_refused / in_tested : 0.0);
  if (!in_misses.empty()) {
    std::printf("  -- first 12 refusals --\n");
    for (size_t i = 0; i < in_misses.size() && i < 12; ++i)
      std::printf("%s\n", in_misses[i].c_str());
  }

  // ---- property 2: out-of-scope requests must not be acted on ----
  std::printf("\n-- out-of-scope sentences: a tool call is a miss --\n");
  int oos_tested = 0, oos_acted = 0;
  std::vector<std::string> oos_misses;
  for (const char* split : {"oos", "oos_val", "oos_test", "val_oos", "test_oos"}) {
    const auto it = by_intent.find(split);
    if (it == by_intent.end()) continue;
    std::printf("  split \"%s\": %zu sentences\n", split, it->second.size());
    for (int k = 0; k < 300 && k < (int)it->second.size(); ++k) {
      const std::string& q = it->second[k];
      std::string reply; float sim = 0;
      if (am_query_public(am, q, reply, sim) != 0) continue;
      oos_tested++;
      if (reply.find("<|tool|>") != std::string::npos) {
        oos_acted++;
        oos_misses.push_back("    \"" + q + "\"");
      }
    }
  }
  std::printf("\n  tested %d out-of-scope sentences, %d acted on (%.1f%%)\n",
              oos_tested, oos_acted,
              oos_tested ? 100.0 * oos_acted / oos_tested : 0.0);
  if (!oos_misses.empty()) {
    std::printf("  -- first 12 actions --\n");
    for (size_t i = 0; i < oos_misses.size() && i < 12; ++i)
      std::printf("%s\n", oos_misses[i].c_str());
  }

  // ---- property 3: no fabricated payloads ----
  //
  // Added because MUTATION TESTING found this unguarded: switching the command
  // vocabulary off, or letting a bare word count as a file reference, changed nothing
  // that any suite noticed. Both licences were implemented (defect 109) and neither
  // was tested -- the kind of gap that reading the code cannot reveal, because the
  // code looks complete.
  //
  // Each case below is a sentence whose LAST WORD is not an argument for anything.
  // Before the licences existed these produced run_shell {"cmd":"low"},
  // read_file {"path":"notes"}, and similar; a correct system refuses.
  std::printf("\n-- fabricated payloads: must be refused --\n");
  int fab_tested = 0, fab_acted = 0;
  {
    static const char* kFab[] = {
      "set a warning for when my bank account starts running low",
      "what'll an education at boulder state college run",
      "can you give me the standard tuning notes",
      "can you give me the most current market news",
      "read text",
      // ---------------------------------------------------------------------
      //  THE BARE-WORD CASES, which the first five did not actually reach.
      //
      //  Mutation testing caught this: allowing a bare word to count as a file
      //  reference (has_extension -> true) changed nothing, because the five
      //  sentences above were already stopped earlier -- by the command licence or by
      //  intent inference -- long before the file-reference test ran. They proved
      //  something, just not the thing the mutation tested.
      //
      //  These three name a read verb EXPLICITLY, so the intent is ReadFile and the
      //  only remaining question is whether the payload is a file reference. The
      //  payload is a bare English word, which is not.
      // ---------------------------------------------------------------------
      "read the file build",
      "open the file config",
      "show me the log",
    };
    for (const char* q : kFab) {
      std::string reply; float sim = 0;
      if (am_query_public(am, q, reply, sim) != 0) continue;
      fab_tested++;
      if (reply.find("<|tool|>") != std::string::npos) {
        fab_acted++;
        std::printf("    ACTED: \"%s\"\n", q);
      }
    }
    std::printf("  %d of %d sentences about the world produced a tool call\n",
                fab_acted, fab_tested);
  }

  // ---- verdict ----
  //
  // ASSERTED NOW, because the first run established the numbers and these are read
  // off it rather than guessed. The two thresholds have different characters:
  //
  //   OUT-OF-SCOPE, which must be tight. Acting on a request this system cannot
  //   serve is the unrecoverable direction: a refusal costs the user one rephrase,
  //   a wrong action writes a file or runs a command. Measured 9.0% after the
  //   confidence floor moved to the shared layer, against 92.0% before -- the
  //   ceiling is set at 15% so a regression to anything like the old behaviour
  //   fails loudly, while ordinary drift does not.
  //
  //   IN-SCOPE, which must be loose. The measured floor is 18.0% (9 of 50), and
  //   most of those are legitimate: CLINC150's calculator intent contains requests
  //   for derivatives, factoring and "solve this math problem", none of which is
  //   arithmetic on two operands. A tight bound here would punish correct refusals
  //   -- defect 53's error. The ceiling at 40% catches a collapse without
  //   misreporting design as failure.
  std::printf("\n-- verdict --\n");
  int exit_code = 0;

  if (oos_tested > 0) {
    const double rate = 100.0 * oos_acted / oos_tested;
    std::printf("  out-of-scope acted on : %.1f%%  (ceiling 15.0%%)\n", rate);
    if (rate > 15.0) {
      std::printf("  FAIL: the system is acting on requests it cannot serve.\n"
                  "  This is the direction that cannot be undone, so the bound is tight.\n");
      exit_code = 1;
    }
  } else {
    std::printf("  FAIL: no out-of-scope sentences were tested -- the oos split was\n"
                "  not found, so the more important property went unmeasured.\n");
    exit_code = 1;
  }

  if (fab_tested == 0) {
    std::printf("  FAIL: the fabricated-payload probe ran nothing.\n");
    exit_code = 1;
  } else if (fab_acted != 0) {
    std::printf("  FAIL: %d of %d sentences yielded a tool call with a payload taken\n"
                "  from the sentence's last word. A fabricated argument is worse than a\n"
                "  refusal because it looks like an answer.\n", fab_acted, fab_tested);
    exit_code = 1;
  } else {
    std::printf("  fabricated payloads    : none (0 of %d)\n", fab_tested);
  }

  if (in_tested > 0) {
    const double rate = 100.0 * in_refused / in_tested;
    std::printf("  in-scope refused      : %.1f%%  (ceiling 40.0%%)\n", rate);
    if (rate > 40.0) {
      std::printf("  FAIL: most in-scope requests are being refused.\n");
      exit_code = 1;
    }
  }

  std::printf("\n  %s\n", exit_code ? "FAILED" : "ALL PASS");
  return exit_code;
}

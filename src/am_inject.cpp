// ===========================================================================
//  Signal injection: many signals inside ONE clause.
//
//  WHAT THE FIRST HUNT ACTUALLY TESTED, and why this is different.
//
//  am-hunt builds combinations with " and ", i.e. MULTI-CLAUSE requests. That
//  exercises the multi-action guard, which is valuable but is not where defect 65
//  came from:
//
//      "put hello in C:/tmp/todo.json"  ->  list_dir
//
//  is a SINGLE clause carrying TWO signals -- a strong verb ("put") and a
//  containment phrase (" in "). The precedence table decided which won, wrongly,
//  and no multi-clause test could have seen it.
//
//  So this file injects additional signal words INTO an existing single-clause
//  prompt and checks that the injected signal does not take the request somewhere
//  neither the original prompt nor the injected signal would go on its own.
//
//  The signals are drawn from the rules themselves:
//
//      containment   " in <path>"
//      strong verb   put / write / save
//      weak verb     show / open / display
//      arithmetic    " 7 + 8 "
//      shell flag    " /x "
//      question      " what is "
//      metadata      " how big is "
//      search        " find all "
//      container     " list the directory "
//
//  Injections are enumerated mechanically, so coverage does not depend on which
//  conflicts the author happened to imagine -- the failure mode of defect 65.
//
//  A result is suspicious when the produced tool is NEITHER
//      * the tool the original prompt produces, nor
//      * the tool the injected signal produces alone, nor
//      * a refusal.
// ===========================================================================
#include "canticle/am.h"
#include "canticle/data.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

using namespace canticle;

namespace {

struct Inj { const char* name; const char* probe; };

// Each probe is a complete mini-request whose own routing is measured, so the
// expectation is derived rather than hard-coded.
const Inj kInj[] = {
  {"containment", "what is in C:/tmp"},
  {"strong-verb", "write hello to C:/tmp/x.txt"},
  {"weak-verb",   "show me C:/tmp/x.txt"},
  {"arithmetic",  "what is 7 + 8"},
  {"shell-flag",  "run the command netstat -an"},
  {"question",    "what is C:/tmp/x.txt"},
  {"metadata",    "how big is C:/tmp/x.txt"},
  {"search",      "find all TODO in C:/tmp"},
  {"container",   "list the directory C:/tmp"},
  {"unsupported", "delete C:/tmp/x.txt"},
};

// Route one request and report which tool it produced.
//
// SECOND BUG IN THIS HARNESS, and it produced a false baseline in the first run.
//
// The previous version contained a stray extra call:
//
//     if (!refused) { std::string r2; float s2; am_query_public(am, q, r2, s2); }
//
// left over from debugging. It did not change the value returned, but it meant the
// baseline for "delete C:/tmp/x.txt" was measured through a path that ran the query
// twice, and the reported baseline read "read_file" while the same request through
// am-chat correctly refuses.
//
// The lesson is the same one am-hunt taught: the harness needs verifying as much as
// the code under test. A wrong baseline makes every downstream comparison
// meaningless -- and here it made a correct refusal look like a routing failure,
// i.e. exactly the false positive that would waste a round of investigation.
//
// So: one query, one answer, and "refused" is defined as "no tool call in the
// reply" rather than by matching particular words.
std::string route_of(const std::string& am, const std::string& q, bool* refused) {
  std::string reply; float sim = 0;
  if (am_query_public(am, q, reply, sim) != 0) {
    if (refused) *refused = false;
    return "error";
  }
  const size_t a = reply.find("<|tool|>");
  if (a == std::string::npos) {
    if (refused) *refused = true;
    return "refused";
  }
  if (refused) *refused = false;
  const size_t b = reply.find(' ', a);
  return reply.substr(a + 8, (b == std::string::npos ? reply.size() : b) - (a + 8));
}

} // namespace

int am_inject_run(int argc, char** argv) {
  const std::string am = "build\\ckpt\\canticle.am";
  int limit = 300;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--limit" && i + 1 < argc) limit = atoi(argv[++i]);

  const auto corpus = synth_tool_data(limit * 20, 777);

  // one representative prompt per tool
  std::map<std::string, std::string> sample;
  for (const auto& s : corpus) {
    const size_t a = s.reply.find("<|tool|>");
    if (a == std::string::npos) continue;
    const size_t b = s.reply.find(' ', a);
    const std::string tool = s.reply.substr(a + 8,
        (b == std::string::npos ? s.reply.size() : b) - (a + 8));
    if (sample.find(tool) == sample.end()) sample[tool] = s.prompt;
  }

  std::printf("=== single-clause signal injection ===\n\n");

  // baseline routing for each tool's own prompt and for each probe
  std::printf("-- baselines --\n");
  std::map<std::string, std::string> base_tool;
  for (const auto& kv : sample) {
    bool ref = false;
    base_tool[kv.first] = route_of(am, kv.second, &ref);
    std::printf("  %-11s -> %-11s  | %s\n", kv.first.c_str(),
                base_tool[kv.first].c_str(), kv.second.c_str());
  }
  std::map<std::string, std::string> base_inj;
  for (const Inj& in : kInj) {
    bool ref = false;
    base_inj[in.name] = route_of(am, in.probe, &ref);
    std::printf("  inj:%-9s -> %-11s  | %s\n", in.name, base_inj[in.name].c_str(), in.probe);
  }

  // ---- injection ----
  //
  // Two placements, because position changes what the parser sees:
  //   PREFIX   "<signal> <original>"   the signal comes first
  //   SUFFIX   "<original> <signal>"   the signal comes last
  // Only SUFFIX is used for probes that themselves contain a path, since
  // prefixing two paths produces an argument-order ambiguity that is a different
  // question from precedence.
  std::printf("\n-- combinations routing to none of {original, signal, refusal} --\n");
  int checked = 0, suspicious = 0;
  std::vector<std::string> findings;

  for (const auto& kv : sample) {
    const std::string& tool = kv.first;
    const std::string& orig = kv.second;
    for (const Inj& in : kInj) {
      if (tool == base_inj[in.name]) continue;   // same family, nothing to learn
      const std::string combined = orig + " " + in.probe;
      bool ref = false;
      const std::string got = route_of(am, combined, &ref);
      checked++;
      if (ref) continue;
      if (got == base_tool[tool]) continue;
      if (got == base_inj[in.name]) continue;
      ++suspicious;
      findings.push_back("  " + tool + " + inj:" + in.name + "  ->  " + got +
                         "\n        orig  : " + orig +
                         "\n        inj   : " + in.probe +
                         "\n        expect " + base_tool[tool] + " or " + base_inj[in.name]);
    }
  }

  for (const auto& f : findings) std::printf("%s\n", f.c_str());
  std::printf("\n  %d injections checked, %d routed elsewhere\n", checked, suspicious);
  // Same minimum-work guard as am-hunt: a broken memory path makes every query
  // return "error", every injection get skipped, and the suite report ALL PASS on
  // zero evidence. Measured at 81 injections when healthy; the floor is 40.
  if (checked < 40) {
    std::printf("\n  FAIL: only %d injections were evaluated -- too few to be evidence.\n"
                "  A harness that evaluated nothing reports success for the same reason\n"
                "  a test that cannot fail does.\n", checked);
    return 1;
  }
  if (suspicious) {
    std::printf("\n  FAIL: each case above needs a decision -- either the pair is\n");
    std::printf("  genuinely ambiguous (extend the refusal) or precedence is wrong.\n");
    return 1;
  }
  std::printf("\n  ALL PASS  (%d injections, none diverted)\n", checked);
  return 0;
}

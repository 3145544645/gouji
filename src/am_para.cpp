
// ===========================================================================
//  Paraphrase robustness harness.
//
//  Gap A: the encoder is pure character n-gram hashing, so it has NO notion of
//  meaning. "read the file X" and "show me X" share almost no trigrams, so the
//  retriever treats them as unrelated.
//
//  Before designing a fix, measure how bad it actually is. This file holds a
//  hand-written paraphrase set with the tool each sentence SHOULD trigger, and
//  reports routing accuracy. No training data is reused: every sentence here is
//  a phrasing the memory has never stored.
// ===========================================================================
#include "canticle/am.h"
#include "canticle/ckpt.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

namespace canticle {

struct ParaCase { const char* text; const char* expect_tool; const char* expect_payload; };

// Each group exercises one intent with phrasings that a keyword matcher fails on.
static const ParaCase kPara[] = {
  // ---- read ----
  {"read the file C:/work/gamma.txt",   "read_file",  "C:/work/gamma.txt"},
  {"show me C:/work/gamma.txt",         "read_file",  "C:/work/gamma.txt"},
  {"cat C:/work/gamma.txt",             "read_file",  "C:/work/gamma.txt"},
  {"open C:/work/gamma.txt",            "read_file",  "C:/work/gamma.txt"},
  {"display the contents of C:/work/a.md", "read_file", "C:/work/a.md"},
  {"fetch C:/work/notes.json",          "read_file",  "C:/work/notes.json"},
  {"let me see C:/work/log.txt",        "read_file",  "C:/work/log.txt"},
  {"打印 C:/work/gamma.txt 的内容",      "read_file",  "C:/work/gamma.txt"},
  // ---- list ----
  {"list the directory C:/tmp",         "list_dir",   "C:/tmp"},
  {"ls C:/tmp",                         "list_dir",   "C:/tmp"},
  {"what is in C:/tmp",                 "list_dir",   "C:/tmp"},
  {"show the entries of C:/tmp",        "list_dir",   "C:/tmp"},
  {"enumerate files under C:/tmp",      "list_dir",   "C:/tmp"},
  {"列出 C:/tmp 下的文件",               "list_dir",   "C:/tmp"},
  // ---- write ----
  {"write hello to D:/data/x.txt",      "write_file", "D:/data/x.txt"},
  {"save hello into D:/data/x.txt",     "write_file", "D:/data/x.txt"},
  {"create D:/data/x.txt with hello",   "write_file", "D:/data/x.txt"},
  {"put hello in D:/data/x.txt",        "write_file", "D:/data/x.txt"},
  {"写入 hello 到 D:/data/x.txt",        "write_file", "D:/data/x.txt"},
  // ---- shell ----
  {"run the command hostname",          "run_shell",  "hostname"},
  {"execute hostname",                  "run_shell",  "hostname"},
  {"call hostname",                     "run_shell",  "hostname"},
  {"please run whoami",                 "run_shell",  "whoami"},
  {"执行 Get-Date",                      "run_shell",  "Get-Date"},
  // ---- calc ----
  {"what is 12 * 7",                    "calc",       "12*7"},
  {"compute 12 * 7",                    "calc",       "12*7"},
  {"calculate 12 * 7",                  "calc",       "12*7"},
  {"12 * 7 = ?",                        "calc",       "12*7"},
  {"how much is 12 * 7",                "calc",       "12*7"},
  {"算一下 12 * 7",                      "calc",       "12*7"},
  // ---------------------------------------------------------------------------
  //  THE TWO-ARGUMENT TOOLS, whose payload transfer needs a dedicated rewrite.
  //
  //  Mutation testing found this gap: disabling rewrite_multi_arg changed no verdict,
  //  because no paraphrase case exercised copy or search. Both were covered in the
  //  corpus, unit-tested at the extractor level, and end-to-end verified by am-eval --
  //  but not by this suite, which is the one that would have caught a regression in
  //  the rewrite specifically.
  //
  //  The payload checked is the DESTINATION (copy) and the NEEDLE (search), because
  //  those are the fields the stored template would otherwise supply -- a regression
  //  leaves the template's value in place, so asserting the user's value is exactly
  //  the right test.
  // ---------------------------------------------------------------------------
  {"copy C:/log/a.txt to C:/work/b.txt",      "copy",   "C:/work/b.txt"},
  {"duplicate C:/log/a.txt as C:/work/b.txt", "copy",   "C:/work/b.txt"},
  {"find all TODO in C:/Users",               "search", "TODO"},
  {"locate FIXME in D:/repo",                 "search", "FIXME"},
};

static std::string tool_of(const std::string& reply) {
  size_t a = reply.find("<|tool|>");
  if (a == std::string::npos) return "(chat)";
  size_t b = reply.find(' ', a);
  return reply.substr(a + 8, (b == std::string::npos ? reply.size() : b) - (a + 8));
}

} // namespace canticle

using namespace canticle;

int am_para_run(int argc, char** argv) {
  std::string ampath = "build\\ckpt\\canticle.am";
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--am" && i + 1 < argc) ampath = argv[++i];

  const int N = (int)(sizeof(kPara) / sizeof(kPara[0]));
  int ok = 0;
  std::printf("=== paraphrase robustness (gap A) ===\n");
  std::printf("%d phrasings, none of them in the memory\n\n", N);

  std::printf("--- intent inference alone (no retrieval) ---\n");
  int iok = 0;
  for (int i = 0; i < N; ++i) {
    const Intent in = infer_intent(kPara[i].text);
    const char* got = intent_name(in);
    const bool hit = !std::strcmp(got, kPara[i].expect_tool);
    if (hit) iok++;
    if (!hit) std::printf("  MISS  %-38s want %-10s inferred %s\n",
                          kPara[i].text, kPara[i].expect_tool, got);
  }
  std::printf("  intent accuracy: %d/%d  %.1f%%\n\n", iok, N, 100.0 * iok / N);
  std::printf("--- retrieval + payload transfer ---\n");

  struct Group { const char* tool; int hit, tot; };
  std::vector<Group> groups;
  auto group_of = [&](const char* t) -> Group* {
    for (auto& g : groups) if (!std::strcmp(g.tool, t)) return &g;
    groups.push_back({t, 0, 0});
    return &groups.back();
  };

  for (int i = 0; i < N; ++i) {
    std::string reply; float sim = 0;
    if (am_query_public(ampath, kPara[i].text, reply, sim) != 0) {
      std::printf("  [ERR] %s\n", kPara[i].text);
      continue;
    }
    const std::string got = tool_of(reply);
    const bool tool_ok = (got == kPara[i].expect_tool);
    // Payload matters as much as routing: a correct tool with the wrong path
    // still reads the wrong file. Compare against the payload carried in the
    // reply, whitespace-insensitively for expressions.
    const std::string want_p = kPara[i].expect_payload;
    const bool payload_ok = (reply.find(want_p) != std::string::npos);
    const bool hit = tool_ok && payload_ok;
    if (hit) ok++;
    Group* g = group_of(kPara[i].expect_tool);
    g->tot++; if (hit) g->hit++;
    std::printf("  %s  %-34s want %-10s got %-10s payload %s  sim=%.3f\n",
                hit ? "[ok]  " : "[MISS]", kPara[i].text,
                kPara[i].expect_tool, got.c_str(),
                payload_ok ? "ok" : "WRONG", sim);
    if (!hit && tool_ok && !payload_ok)
      std::printf("         reply was: %s\n", reply.c_str());
  }

  std::printf("\n--- per intent ---\n");
  for (auto& g : groups)
    std::printf("  %-12s %2d/%2d  %5.1f%%\n", g.tool, g.hit, g.tot,
                100.0 * g.hit / std::max(1, g.tot));
  std::printf("\n  OVERALL  %d/%d  %.1f%%\n", ok, N, 100.0 * ok / N);
  return ok == N ? 0 : 1;
}

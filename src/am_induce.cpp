// ===========================================================================
//  Frame induction tests.
//
//  WHAT IS BEING CHECKED
//  ---------------------
//  That several observations sharing a surface frame collapse into ONE canonical
//  pattern, and that unrelated observations do NOT collapse into a bogus one.
//
//  The negative case carries most of the weight. An over-eager frame is worse
//  than a missing one: a frame matches by prefix, so a wrong frame silently
//  captures requests that belong to other tools. Two requests sharing only a
//  stopword must never produce a rule.
// ===========================================================================
#include "canticle/induce.h"
#include <cstdio>
#include <string>
#include <vector>

using namespace canticle;

static int failures = 0;

static void check(bool cond, const char* what) {
  std::printf("  %s %s\n", cond ? "[ok]  " : "[FAIL]", what);
  if (!cond) failures++;
}

int am_induce_run(int, char**) {
  std::printf("=== frame induction ===\n\n");

  // ---- POSITIVE: three observations, one varying payload ----
  {
    std::vector<std::pair<std::string,std::string>> obs = {
      {"play some jazz",  "<|tool|>run_shell {\"cmd\":\"jazz\"}<|/tool|>"},
      {"play some blues", "<|tool|>run_shell {\"cmd\":\"blues\"}<|/tool|>"},
      {"play some metal", "<|tool|>run_shell {\"cmd\":\"metal\"}<|/tool|>"},
    };
    const auto f = induce_frames(obs);
    std::printf("--- three paraphrases ---\n");
    for (const auto& x : f)
      std::printf("    frame \"%s\" from %d obs, %zu payloads -> %s\n",
                  x.text.c_str(), x.observations, x.payloads.size(),
                  x.action_template.c_str());
    check(f.size() == 1, "exactly one frame induced");
    if (!f.empty()) {
      check(f[0].text == "play some {}", "frame keeps the two shared tokens");
      check(f[0].payloads.size() == 3, "all three payloads collected");
    }
  }

  // ---- NEGATIVE: repetition of ONE example is not a pattern ----
  {
    std::vector<std::pair<std::string,std::string>> obs = {
      {"play some jazz", "<|tool|>run_shell {\"cmd\":\"jazz\"}<|/tool|>"},
      {"play some jazz", "<|tool|>run_shell {\"cmd\":\"jazz\"}<|/tool|>"},
      {"play some jazz", "<|tool|>run_shell {\"cmd\":\"jazz\"}<|/tool|>"},
    };
    const auto f = induce_frames(obs);
    std::printf("--- one example repeated three times ---\n");
    check(f.empty(), "a repeated example yields no frame (needs distinct payloads)");
  }

  // ---- NEGATIVE: unrelated requests must not collapse ----
  {
    std::vector<std::pair<std::string,std::string>> obs = {
      {"read the file alpha.txt",  "<|tool|>read_file {\"path\":\"alpha.txt\"}<|/tool|>"},
      {"compute 12 + 7",           "<|tool|>calc {\"expr\":\"12+7\"}<|/tool|>"},
      {"list the directory C:/tmp","<|tool|>list_dir {\"path\":\"C:/tmp\"}<|/tool|>"},
    };
    const auto f = induce_frames(obs);
    std::printf("--- three unrelated requests ---\n");
    for (const auto& x : f)
      std::printf("    (unexpected) frame \"%s\"\n", x.text.c_str());
    check(f.empty(), "unrelated requests produce no frame");
  }

  // ---- NEGATIVE: one shared token is not structure ----
  {
    std::vector<std::pair<std::string,std::string>> obs = {
      {"read the file alpha",  "{\"path\":\"a\"}"},
      {"read the thing beta",  "{\"path\":\"b\"}"},
      {"read some data gamma", "{\"path\":\"c\"}"},
    };
    const auto f = induce_frames(obs);
    std::printf("--- only one shared token (read) ---\n");
    for (const auto& x : f)
      std::printf("    (unexpected) frame \"%s\"\n", x.text.c_str());
    check(f.empty(), "a single shared token is below MIN_FRAME_TOKENS");
  }

  // ---- frame_matches: what a frame claims ----
  //
  // This predicate decides which taught examples get suppressed in favour of the
  // frame row, so a false positive silently deletes an example and a false
  // negative leaves the memory larger than it needs to be.
  {
    Frame f;
    f.text = "play some {}";
    std::printf("--- frame membership ---\n");
    check(frame_matches(f, "play some jazz"),     "member: new payload");
    check(frame_matches(f, "please play some jazz"), "member: extra prefix token allowed");
    check(!frame_matches(f, "play some"),          "non-member: no payload to substitute");
    check(!frame_matches(f, "play jazz"),          "non-member: missing a frame token");
    check(!frame_matches(f, "read the file"),      "non-member: unrelated");
  }

  std::printf("\n  %s  (%d failure(s))\n", failures ? "FAILED" : "ALL PASS", failures);
  return failures ? 1 : 0;
}

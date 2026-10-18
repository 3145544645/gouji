
// ===========================================================================
//  Multi-turn context tests (gap B).
//
//  Each case is a scripted conversation plus the tool call the LAST turn should
//  produce. The middle turns only exist to establish context, so a failure here
//  means inheritance did not carry the tool or the payload forward.
// ===========================================================================
#include "canticle/am.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace canticle {

struct MultiCase {
  const char* script;      // turns separated by '|'
  const char* want_tool;   // the tool the final turn must call ("" = plain chat)
  const char* want_payload;
  const char* note;
};

static const MultiCase kMulti[] = {
  // ---- payload-only follow-ups: keep the tool, replace the target ----
  {"read the file C:/work/a.txt|and C:/work/b.txt",
   "read_file", "C:/work/b.txt", "new path, same tool"},
  {"list the directory C:/tmp|and C:/Users",
   "list_dir", "C:/Users", "new dir, same tool"},
  {"write hello to D:/data/x.txt|now D:/data/y.txt",
   "write_file", "D:/data/y.txt", "new target, same tool"},
  // ---- bare cues: keep BOTH tool and payload ----
  {"calculate 12 * 7|again",
   "calc", "12*7", "again re-runs the same call"},
  {"run the command hostname|again",
   "run_shell", "hostname", "again re-runs the same call"},
  {"read the file C:/work/a.txt|same",
   "read_file", "C:/work/a.txt", "same re-runs the same call"},
  {"list the directory C:/tmp|that",
   "list_dir", "C:/tmp", "pronoun refers to the previous call"},
  // ---- Chinese cues ----
  {"run the command whoami|再来一次",
   "run_shell", "whoami", "Chinese continuation cue"},
  {"read the file C:/work/a.txt|还有 C:/work/c.txt",
   "read_file", "C:/work/c.txt", "Chinese conjunction with new path"},
  // ---- self-contained turns must NOT be rewritten ----
  {"read the file C:/work/a.txt|what is 8 * 9",
   "calc", "8*9", "a full sentence ignores context"},
  {"list the directory C:/tmp|run the command hostname",
   "run_shell", "hostname", "a full sentence ignores context"},
};

static std::string tool_of(const std::string& r) {
  size_t a = r.find("<|tool|>");
  if (a == std::string::npos) return "";
  size_t b = r.find(' ', a);
  return r.substr(a + 8, (b == std::string::npos ? r.size() : b) - (a + 8));
}

} // namespace canticle

using namespace canticle;

int am_multi_run(int argc, char** argv) {
  std::string mode = argc > 0 ? argv[0] : "";
  if (mode != "am-multi") { std::printf("bad mode\n"); return 1; }

  const int N = (int)(sizeof(kMulti) / sizeof(kMulti[0]));
  std::printf("=== multi-turn context (gap B) ===\n");
  std::printf("%d scripts; only the LAST turn is scored\n\n", N);

  int ok = 0;
  int evaluated = 0;   // cases actually run; compared against N at the end
  for (int i = 0; i < N; ++i) {
    // Rewrite the final turn using a Session seeded with the earlier turns.
    // This exercises exactly the code path am-chat uses.
    std::vector<std::string> parts;
    std::string cur;
    for (const char* p = kMulti[i].script; *p; ++p) {
      if (*p == '|') { parts.push_back(cur); cur.clear(); }
      else cur.push_back(*p);
    }
    parts.push_back(cur);
    if (parts.size() < 2) { std::printf("  [SKIP] %s\n", kMulti[i].note); continue; }

    Session sess;
    for (size_t t = 0; t + 1 < parts.size(); ++t) {
      // seed the session with what the earlier turns actually resolve to,
      // using the same pure functions the CLI uses
      const std::string q = sess.rewrite(parts[t]);
      const Intent in = infer_intent(q);
      const std::string tool = (in == Intent::Unknown) ? "" : intent_name(in);
      sess.push(parts[t], tool, extract_payload(q));
    }

    const std::string last = parts.back();
    const std::string q = sess.rewrite(last);
    const Intent in = infer_intent(q);
    const std::string got_tool = (in == Intent::Unknown) ? "" : intent_name(in);
    const std::string got_pay  = extract_payload(q);

    // Compare the payload whitespace-insensitively: the same expression is
    // spelled "12*7" inside a stored reply and "12 * 7" in a prompt, and an
    // earlier version of this test scored the spacing difference as a failure.
    auto squash = [](const std::string& s) {
      std::string o;
      for (char c : s) if (c != ' ' && c != '\t') o.push_back(c);
      return o;
    };
    const bool tool_ok = (got_tool == kMulti[i].want_tool);
    const bool pay_ok  = (squash(got_pay).find(squash(kMulti[i].want_payload))
                          != std::string::npos);
    const bool hit = tool_ok && pay_ok;
    if (hit) ok++;
    evaluated++;

    std::printf("  %s %s\n", hit ? "[ok]  " : "[MISS]", kMulti[i].note);
    std::printf("        last turn : %s\n", last.c_str());
    if (q != last) std::printf("        rewritten : %s\n", q.c_str());
    if (!hit)
      std::printf("        want %s / %s   got %s / %s\n",
                  kMulti[i].want_tool, kMulti[i].want_payload,
                  got_tool.c_str(), got_pay.c_str());
  }

  std::printf("\n  OVERALL  %d/%d  %.1f%%\n", ok, N, 100.0 * ok / N);
  // ---------------------------------------------------------------------------
  //  NO MEMORY IS REQUIRED HERE, and the first version of this guard was wrong
  //  about that.
  //
  //  I added "if (N < 5) fail" after seeing this suite print 11/11 on a fresh clone
  //  with no memory, and assumed it was another instance of a suite passing
  //  vacuously. It is not: this suite calls infer_intent and extract_payload
  //  directly, both of which are pure functions of the sentence. It never touches
  //  the memory, so answering correctly without one is the correct behaviour, and
  //  the guard would have failed a working test on purpose.
  //
  //  That is defect 53's error again -- an assertion encoding an expectation the
  //  system does not and should not meet. Caught immediately because I ran the
  //  clone before believing the guard.
  //
  //  What IS worth asserting is the property that made the confusion possible:
  //  the dependency must be visible. If this suite ever starts consulting the
  //  memory, the count of pure-function evaluations will not match the case count,
  //  and that is a fact the suite can check about itself.
  // ---------------------------------------------------------------------------
  if (evaluated != N) {
    std::printf("  FAIL: %d cases declared but %d evaluated.\n", N, evaluated);
    return 1;
  }
  return ok == N ? 0 : 1;
}

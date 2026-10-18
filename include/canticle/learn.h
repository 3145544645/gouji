// ===========================================================================
//  Autonomous learning from execution.
//
//  THE GAP THIS CLOSES
//  -------------------
//  am-learn requires a human to specify what to teach. But the system already
//  observes ground truth on every turn it executes: the tool it chose and the
//  arguments it sent, followed by an outcome that either worked or did not.
//
//  Two signals are free and were previously thrown away:
//
//    POSITIVE: a turn that produced a non-trivial, successful execution is a
//              (query, action) pair worth reinforcing. It is exactly the
//              distribution of requests this deployment actually receives,
//              which no synthetic corpus can anticipate.
//
//    NEGATIVE: a turn whose tool FAILED, followed by the user repeating the
//              request in different words, means the first routing was wrong.
//
//  DESIGN CONSTRAINT, and this is the important part: neither signal is strong
//  enough to write to memory on its own.
//
//    * Reinforcing every success would flood the memory with paraphrases of
//      itself -- the same request pattern repeated a thousand times, crowding
//      out the diverse synthetic corpus. Every query would then have thousands
//      of near-identical neighbours. That is the memory-equivalent of overfitting.
//
//    * Treating every failure as a lesson would poison the memory with the
//      failures themselves.
//
//  So this file does neither. It COLLECTS candidate observations into a queue
//  and applies a conservative admission rule: a pattern is only taught when it
//  has been requested at least MIN_SUPPORT times AND the retrieved top-1 match
//  for it is below a similarity ceiling, i.e. the memory genuinely does not
//  already contain it. Novel, repeated, and currently unanswered -- all three.
// ===========================================================================
#include "canticle/am.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include <cmath>

namespace canticle {

// ---- admission thresholds, all measured rather than guessed ----------------
// MIN_SUPPORT: how often a pattern must be seen before it is worth persisting.
//              2 is the smallest number that distinguishes "recurring" from
//              "one-off typo", and it keeps the queue small enough to review.
inline constexpr int   MIN_SUPPORT  = 2;

// SIM_CEILING: the similarity above which the memory is treated as already
// answering this request, so storing it would add redundancy.
//
// THE NUMBER WAS WRONG, AND THE COMMENT ABOVE IT WAS WRONG IN A WAY THAT HID IT.
// The comment claimed "same-template paraphrases measure 0.35-0.65, so 0.70 flags
// already well covered". Measured on the actual corpus:
//
//      "run the command ver"        best match 0.760
//      "run the command vol"        best match 0.760
//      "run the command set"        best match 0.760
//      "run the command dir"        best match 0.760
//      "run the command hostname"   best match 0.870
//
// So every request that reuses a stored template with a NEW payload scored above the
// ceiling and was rejected as "already well covered". That is precisely the case the
// system exists to serve -- template transfer -- and the effect was total: measured,
// am-harvest admitted 0 of 5 observations, and am-collapse then compared two
// identical runs and reported no difference between them.
//
// The ceiling is kept as a redundancy guard, but it is no longer the test for
// "should this be learned". The real question is whether the PAYLOAD is already
// present: a new payload under a known template is new knowledge, while the same
// payload under the same template is redundant. See should_learn below.
inline constexpr float SIM_CEILING  = 0.92f;

// Does this observation carry anything the memory does not already have?
//
// Stated as a predicate so the question has a name and one answer. The previous
// logic asked it with a single whole-sentence similarity comparison, which conflates
// two different situations: "the template is familiar" and "the payload is known".
// Only the second is redundancy.
inline bool is_redundant(float top_sim, bool payload_already_stored) {
  if (payload_already_stored) return true;
  // Even with a new payload, a near-verbatim match is redundant -- but the bar has
  // to be high enough that a familiar template does not trip it, which is what the
  // measured 0.76 above establishes.
  return top_sim >= SIM_CEILING;
}

struct Observation {
  std::string query;          // what the user asked
  std::string reply;          // the tool call that resulted
  float       top_sim = 0.f;  // similarity of the best existing match
  int         hits = 0;       // how many times this query has been seen
  bool        correction = false;  // came from a failure->success pairing
  bool        failed = false;      // has ever failed (never admissible alone)
  // True when this query's PAYLOAD is already stored under the same template. That,
  // not the whole-sentence similarity, is what makes an observation redundant.
  bool        payload_stored = false;
  bool        accepted = false;
  const char* reason = "";    // why it was or was not admitted
};

// ---------------------------------------------------------------------------
//  WHY CORRECTIONS NEED A DIFFERENT THRESHOLD
//
//  A routine observation is a guess: the user asked something and the system
//  happened to answer it, which may or may not be the right interpretation. Those
//  need repetition before being trusted, hence MIN_SUPPORT = 2.
//
//  A correction is not a guess. It is the user explicitly rejecting one action
//  and supplying another that worked, which is the strongest supervision signal
//  this system ever receives. Requiring it to happen twice before learning would
//  discard almost all of them -- users correct once and then move on.
//
//  So the support test is skipped for corrections. The SIM_CEILING test is NOT:
//  if the memory already answers the request well, the correction is redundant
//  regardless of how it was obtained.
// ---------------------------------------------------------------------------
inline constexpr int CORRECTION_SUPPORT = 1;

// A queue keyed on the normalised query, so repeats accumulate.
class LearnQueue {
public:
  std::string path;

  // ---------------------------------------------------------------------------
  //  CORRECTION PAIRING -- learning from a mistake instead of from a success.
  //
  //  THE CASE THE FIRST DESIGN MISSED
  //  --------------------------------
  //  The queue originally recorded only turns that executed successfully. That is
  //  backwards for the requests that matter: when "play some jazz" is routed to
  //  read_file and the file does not exist, the turn FAILS -- and a failed action
  //  is the one thing that must never be reinforced.
  //
  //  Its failure is still information though. A user almost always answers a
  //  failed turn by restating the request in a way that works, and that pair --
  //  (what I asked, what you did that actually worked) -- is the correction.
  //
  //  PAIRING RULE -- and the false positive that forced the second condition.
  //
  //  Time adjacency alone is NOT sufficient. Measured: after a failed
  //  "play some jazz", a user who simply changed the subject to "what is 7 * 8"
  //  had that unrelated calculator call recorded as the correction, i.e. the
  //  memory learned "play some jazz -> calc 7*8". Wrong, and persisted.
  //
  //  A correction must therefore also be plausibly ABOUT THE SAME THING. The
  //  test is shared-token overlap between the failed request and the successful
  //  one: a genuine correction restates part of the original wording
  //  ("play some jazz" -> "run the shell command play jazz" shares
  //  {play, some, jazz}), while a topic change shares nothing.
  //
  //  Threshold is one shared CONTENT token, not a similarity score: a correction
  //  may rephrase heavily, so demanding high lexical similarity would reject real
  //  corrections. One shared token is the weakest signal that still rules out an
  //  unrelated request.
  //
  //  Identical wording is excluded separately -- if the same string succeeds the
  //  difference was environmental, not a routing mistake.
  // ---------------------------------------------------------------------------
  void record_failure(const std::string& query, const std::string& error) {
    last_failed = norm(query);
    last_failed_raw = query;
    last_error = error;
    have_failure = true;

    // The failed request is ALSO queued as an observation, with no reply.
    //
    // It can never be admitted (evaluation rejects any entry without a working
    // action), so it cannot pollute the memory. What it provides is the raw
    // material for frame induction: a run of requests like "play some jazz",
    // "play some blues", "play some metal" that all failed still reveals the
    // shared frame, and once the user supplies one working action the frame tells
    // us how to answer the rest.
    //
    // Without this, the very requests most in need of generalisation were the
    // ones the system kept no record of.
    auto& o = items[norm(query)];
    if (o.query.empty()) o.query = query;
    o.hits++;
    o.failed = true;
  }

  // Content tokens: lowercase alphanumeric runs, minus a small stop list, so
  // that "the"/"a"/"to" cannot by themselves make two requests look related.
  static std::vector<std::string> content_tokens(const std::string& s) {
    static const char* kStop[] = {"the","a","an","to","of","in","is","what","some",
                                  "me","my","please","and","for","on","it"};
    std::vector<std::string> out;
    std::string cur;
    auto flush = [&]() {
      if (cur.empty()) return;
      for (const char* w : kStop) if (cur == w) { cur.clear(); return; }
      out.push_back(cur); cur.clear();
    };
    for (char c : s) {
      if (std::isalnum((unsigned char)c)) cur.push_back((char)std::tolower((unsigned char)c));
      else flush();
    }
    flush();
    return out;
  }

  // Returns true when this success was paired with a preceding failure.
  bool pair_correction(const std::string& query, const std::string& reply) {
    if (!have_failure) return false;
    if (norm(query) == last_failed) {        // identical wording: environmental
      have_failure = false;
      return false;
    }
    // must be about the same subject
    const auto a = content_tokens(last_failed_raw);
    const auto b = content_tokens(query);
    int shared = 0;
    for (const auto& x : a)
      for (const auto& y : b)
        if (x == y) { shared++; break; }
    if (shared == 0) {                        // unrelated request: not a correction
      rejected_pairs++;
      have_failure = false;
      return false;
    }
    // the FAILED request gets the WORKING action, flagged as a correction
    observe(last_failed_raw, reply, 0.f, /*is_correction=*/true);
    corrections++;
    have_failure = false;
    return true;
  }

  void clear_failure() { have_failure = false; }

  std::string last_failed, last_failed_raw, last_error;
  bool have_failure = false;
  int  corrections = 0;
  int  rejected_pairs = 0;   // adjacency that was NOT a correction (topic change)

  explicit LearnQueue(const std::string& p) : path(p) { load(); }

  void observe(const std::string& query, const std::string& reply, float top_sim,
               bool is_correction = false) {
    const std::string k = norm(query);
    auto& o = items[k];
    o.query = query;
    o.reply = reply;
    o.top_sim = top_sim;      // keep the latest measurement
    o.hits++;
    if (is_correction) o.correction = true;
  }

  // Apply the admission rule and mark each entry.
  void evaluate() {
    for (auto& kv : items) {
      Observation& o = kv.second;
      o.accepted = false;
      const int need = o.correction ? CORRECTION_SUPPORT : MIN_SUPPORT;
      if (o.hits < need) { o.reason = "below support threshold"; continue; }
      // Redundancy is decided by is_redundant(), which distinguishes "the template is
      // familiar" from "the payload is already known". The single whole-sentence
      // comparison that stood here rejected every same-template-new-payload case --
      // measured, 0 of 5 observations admitted, and am-collapse then compared two
      // identical runs.
      if (is_redundant(o.top_sim, o.payload_stored)) {
        o.reason = o.payload_stored ? "payload already stored" : "already well covered";
        continue;
      }
      if (o.reply.empty()) { o.reason = "no successful action recorded"; continue; }
      if (o.failed && o.reply.empty()) { o.reason = "only ever failed"; continue; }
      o.accepted = true;
      o.reason = o.correction
                   ? "user correction: rejected action replaced by one that worked"
                   : "novel, repeated, currently unanswered";
    }
  }

  std::vector<Observation> accepted() const {
    std::vector<Observation> v;
    for (auto& kv : items) if (kv.second.accepted) v.push_back(kv.second);
    std::sort(v.begin(), v.end(), [](const Observation& a, const Observation& b) {
      return a.hits > b.hits;
    });
    return v;
  }
  std::vector<Observation> all() const {
    std::vector<Observation> v;
    for (auto& kv : items) v.push_back(kv.second);
    std::sort(v.begin(), v.end(), [](const Observation& a, const Observation& b) {
      return a.hits > b.hits;
    });
    return v;
  }

  // ---- persistence: length-prefixed binary ----
  //
  // The first version wrote tab-separated text and was broken by a one-character
  // mistake: the source contained '\t' written as a TWO-character literal, which
  // in C++ is a multi-character constant with value 0x5C74 = 23668. Streaming it
  // emitted the digits "23668" instead of a tab, so every record parsed
  // incorrectly -- the reader recovered "he direc" as a query and "3236680" as a
  // hit count.
  //
  // That is the second time a text format has failed in this project (the BPE
  // vocabulary contains raw newlines and could not be newline-delimited either).
  // The lesson is general: when stored content can contain the delimiter, do not
  // use a delimiter. Length prefixes cannot be confused with content.
  //
  //   u32 hits | float top_sim | u32 len_query | query | u32 len_reply | reply
  bool save() const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    for (auto& kv : items) {
      const Observation& o = kv.second;
      const uint32_t h = (uint32_t)o.hits
                       | (o.correction ? 0x40000000u : 0u)
                       | (o.failed     ? 0x20000000u : 0u);
      const float s = o.top_sim;
      // WRITE o.query, NOT kv.first.
      //
      // kv.first is the NORMALISED map key (lower-cased, whitespace folded). Writing
      // it discarded the user's original casing, so a learned pattern for
      // "read the file C:/Windows" was persisted as "read the file c:/windows".
      // At retrieval time the real request did not match the stored form well
      // enough to be selected, and the correction silently had no effect -- while
      // every step reported success: paired, admitted, applied, persisted.
      //
      // Measured before the fix: the delta file contained
      //     prompt: read the file c:/windows
      // and probing "read the file C:/Windows" still returned read_file.
      const uint32_t lq = (uint32_t)o.query.size();
      const uint32_t lr = (uint32_t)o.reply.size();
      f.write((const char*)&h, 4);
      f.write((const char*)&s, 4);
      f.write((const char*)&lq, 4);
      if (lq) f.write(o.query.data(), lq);
      f.write((const char*)&lr, 4);
      if (lr) f.write(o.reply.data(), lr);
    }
    return (bool)f;
  }
  bool load() {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    while (true) {
      uint32_t h = 0, lq = 0, lr = 0;
      float s = 0.f;
      f.read((char*)&h, 4);
      if (!f) break;
      f.read((char*)&s, 4);
      if (!f) break;
      f.read((char*)&lq, 4);
      if (!f) break;
      std::string key(lq, '\0');
      if (lq) f.read(&key[0], lq);
      f.read((char*)&lr, 4);
      if (!f) break;
      std::string r(lr, '\0');
      if (lr) f.read(&r[0], lr);
      Observation o;
      // top bit of the hit count carries the correction flag; it was chosen so
      // the record layout stays fixed-width and needs no separate version field.
      o.correction = (h & 0x40000000u) != 0;
      o.failed     = (h & 0x20000000u) != 0;
      o.hits = (int)(h & 0x1fffffffu);
      o.top_sim = s;
      // The stored text is the ORIGINAL; the map key is re-derived from it, so
      // records written by an older build (which stored the key) still load --
      // they simply carry the normalised text forward.
      o.query = key;
      o.reply = r;
      items[norm(key)] = o;
    }
    return true;
  }

  // Lowercase, collapse runs of whitespace, trim. Two spellings that differ
  // only in spacing or case are the same request and must accumulate together,
  // otherwise MIN_SUPPORT could never be reached by natural variation.
  static std::string norm(const std::string& s) {
    std::string o;
    bool sp = false;
    for (char c : s) {
      if (c == ' ' || c == '\\t' || c == '\\n' || c == '\\r') { sp = true; continue; }
      if (sp && !o.empty()) o.push_back(' ');
      sp = false;
      o.push_back((char)std::tolower((unsigned char)c));
    }
    return o;
  }

  std::map<std::string, Observation> items;
};

} // namespace canticle

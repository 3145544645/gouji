// ===========================================================================
//  Template induction: generalise from several corrections at once.
//
//  THE LIMITATION THIS ADDRESSES
//  -----------------------------
//  Every learning path so far stores ONE pattern per observation. Teach
//  "play some jazz" and the memory gains a single row. "play some blues" works
//  afterwards -- but only because the SLOT TRANSFER mechanism substitutes the
//  payload at retrieval time, not because anything generalised.
//
//  That has a concrete cost: a deployment where users say twenty things after
//  "play some" stores twenty rows for one rule, and a request phrased with a
//  verb that no stored row contains still fails.
//
//  WHAT INDUCTION ADDS
//  -------------------
//  Given several observed (query, action) pairs that share a surface frame, emit
//  a SINGLE canonical pattern whose payload is the variable part. The memory then
//  holds one row per rule instead of one per example.
//
//  HOW THE FRAME IS FOUND
//  ----------------------
//  Tokenise each query, then find the longest common subsequence of tokens across
//  all of them. The tokens OUTSIDE that subsequence are the variable positions;
//  their union is the payload. Agreement across a minimum number of distinct
//  examples is what separates a real frame from a coincidence.
//
//  Example -- three observations:
//      "play some jazz"   tokens [play, some, jazz]
//      "play some blues"  tokens [play, some, blues]
//      "play some metal"  tokens [play, some, metal]
//  LCS = [play, some]; variable = {jazz, blues, metal}
//  emitted frame: "play some <payload>"  with a payload slot
//
//  GUARD AGAINST OVER-GENERALISATION, and it matters: an LCS of length 0 or 1 is
//  not a frame, it is noise. Two requests sharing only "the" must not produce a
//  rule. MIN_FRAME_TOKENS enforces that, and MIN_SUPPORT_FOR_FRAME requires the
//  same frame to be seen from enough DISTINCT payloads -- three copies of
//  "play some jazz" is one example repeated, not a pattern.
// ===========================================================================
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cctype>
#include <cstdio>

namespace canticle {

// A frame is stored as its literal tokens with a single {} placeholder.
struct Frame {
  std::string  text;                    // e.g. "play some {}"
  std::vector<std::string> payloads;    // distinct payloads seen
  std::string  action_template;         // the action, with the payload substituted
  int          observations = 0;
};

// Longest common subsequence of two token sequences (order-preserving, gaps ok).
inline std::vector<std::string> tokens(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (std::isalnum((unsigned char)c)) cur.push_back((char)std::tolower((unsigned char)c));
    else { if (!cur.empty()) { out.push_back(cur); cur.clear(); } }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

inline std::vector<std::string> lcs(const std::vector<std::string>& a,
                                    const std::vector<std::string>& b) {
  const size_t n = a.size(), m = b.size();
  std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));
  for (size_t i = 1; i <= n; ++i)
    for (size_t j = 1; j <= m; ++j)
      dp[i][j] = (a[i-1] == b[j-1]) ? dp[i-1][j-1] + 1
                                    : std::max(dp[i-1][j], dp[i][j-1]);
  std::vector<std::string> out;
  size_t i = n, j = m;
  while (i > 0 && j > 0) {
    if (a[i-1] == b[j-1]) { out.push_back(a[i-1]); --i; --j; }
    else if (dp[i-1][j] >= dp[i][j-1]) --i;
    else --j;
  }
  std::reverse(out.begin(), out.end());
  return out;
}

// A frame must keep at least this many fixed tokens. One token is not structure:
// "read {}" and "list {}" share only a verb and would collapse into nonsense.
inline constexpr size_t MIN_FRAME_TOKENS = 2;
// And it must be supported by at least this many DIFFERENT payloads, so that
// repetition of one example cannot masquerade as a general rule.
inline constexpr size_t MIN_SUPPORT_FOR_FRAME = 3;

// ---------------------------------------------------------------------------
//  Does a query belong to a frame?
//
//  Used to collapse several taught examples into the single row that represents
//  their family. The frame's fixed tokens must all occur in the query, in order,
//  with at least one token left over as the payload -- otherwise "play some" would
//  claim a query that consists of nothing but "play some", which has no argument
//  to substitute.
// ---------------------------------------------------------------------------
inline bool frame_matches(const Frame& f, const std::string& query) {
  std::string fixed = f.text;
  const size_t brace = fixed.find(" {}");
  if (brace != std::string::npos) fixed = fixed.substr(0, brace);
  const auto ft = tokens(fixed);
  const auto qt = tokens(query);
  if (ft.empty()) return false;
  size_t j = 0;
  for (size_t i = 0; i < qt.size() && j < ft.size(); ++i)
    if (qt[i] == ft[j]) ++j;
  if (j < ft.size()) return false;   // frame not contained
  return qt.size() > ft.size();      // and something remains to be the payload
}

// Build frames from a set of observations.
inline std::vector<Frame> induce_frames(
    const std::vector<std::pair<std::string,std::string>>& obs) {
  std::vector<Frame> frames;

  // Group observations by their first token. A shared frame almost always shares
  // its leading word, and this keeps the pairwise induction below cheap: without
  // it, N observations cost N^2 LCS computations, and the memory can hold 13500.
  std::map<std::string, std::vector<size_t>> by_head;
  std::vector<std::vector<std::string>> toks;
  for (size_t i = 0; i < obs.size(); ++i) {
    auto t = tokens(obs[i].first);
    toks.push_back(t);
    if (!t.empty()) by_head[t[0]].push_back(i);
  }

  for (auto& kv : by_head) {
    const auto& idx = kv.second;
    if (idx.size() < MIN_SUPPORT_FOR_FRAME) continue;

    // Extend the common subsequence across every member of the group.
    auto common = toks[idx[0]];
    for (size_t k = 1; k < idx.size(); ++k) {
      common = lcs(common, toks[idx[k]]);
      if (common.size() < MIN_FRAME_TOKENS) break;
    }
    if (common.size() < MIN_FRAME_TOKENS) continue;

    // Collect the distinct payloads: tokens present in an observation but not in
    // the common subsequence, taken from the position AFTER the frame's tail so
    // that only the governed argument is captured.
    Frame f;
    {
      std::string t;
      for (size_t i = 0; i < common.size(); ++i) {
        if (i) t += ' ';
        t += common[i];
      }
      f.text = t + " {}";
    }
    for (size_t i : idx) {
      for (const auto& w : toks[i]) {
        bool in_frame = false;
        for (const auto& c : common) if (c == w) { in_frame = true; break; }
        if (in_frame) continue;
        bool dup = false;
        for (const auto& p : f.payloads) if (p == w) { dup = true; break; }
        if (!dup) f.payloads.push_back(w);
      }
      f.observations++;
    }
    if (f.payloads.size() < MIN_SUPPORT_FOR_FRAME) continue;   // one payload repeated

    // The action template comes from the first member of the group that HAS a
    // working action.
    //
    // Not necessarily obs[0]: a failed observation carries a query but no reply,
    // and those failed queries are the whole reason a frame is worth inducing --
    // they are the requests the system cannot yet answer. Requiring obs[0] to
    // have a reply meant a group of three failures plus one correction produced
    // no frame at all, which is precisely the case the feature exists for.
    const std::string* src = nullptr;
    for (size_t i : idx)
      if (!obs[i].second.empty()) { src = &obs[i].second; break; }
    if (!src) { continue; }   // no working action anywhere: nothing to generalise INTO

    std::string tmpl = *src;
    // Replace the quoted argument with a slot so the frame governs the payload.
    const size_t quoted = tmpl.find(":\"");
    if (quoted != std::string::npos) {
      const size_t end = tmpl.find('\"', quoted + 2);
      if (end != std::string::npos) tmpl = tmpl.substr(0, quoted + 2) + "{}" + tmpl.substr(end);
    }
    f.action_template = tmpl;

    frames.push_back(std::move(f));
  }
  return frames;
}

} // namespace canticle

// ===========================================================================
//  Semantic structure on REAL data.
//
//  THE QUESTION
//  ------------
//  At v1.6 the learned lexical embedding was built and then withdrawn: it produced
//  no behavioural change on unseen synonyms while costing 5.4 points of held-out
//  accuracy. The stated reason was never that the method was wrong -- it was that
//  the corpus was 4500 sentences of synthetic templates, so any word a real user
//  might say was out of vocabulary and the embedding contributed noise.
//
//  CLINC150 supplies what was missing: 22500 genuine utterances over 150 intents,
//  written by real people. If the diagnosis was right, the same measurement that
//  read ABSENT before should read PRESENT now.
//
//  THE MEASUREMENT IS THE ONE FROM am-sim, UNCHANGED: for each intent group,
//  compare cosine(a,b) within a group against cosine(a,c) across groups, and count
//  how often a cross-group pair beats the worst within-group pair.
//
//  Both representations are measured on the same real sentences so the comparison
//  is direct rather than across corpora:
//      A. character n-gram encoder (what the memory uses today)
//      B. learned lexical embedding trained on CLINC150
// ===========================================================================
#include "canticle/lex.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>

namespace canticle {

// Minimal JSON reader for the CLINC150 layout: an array of [text, label] pairs.
// Written by hand rather than pulled in as a dependency -- the shape is fixed and
// known, so a full parser would be more code than the scan it replaces.
struct ClincItem { std::string text, label; };

inline std::vector<ClincItem> parse_clinc(const std::string& path) {
  std::vector<ClincItem> out;
  std::ifstream f(path, std::ios::binary);
  if (!f) return out;
  std::stringstream ss; ss << f.rdbuf();
  const std::string s = ss.str();

  // ---------------------------------------------------------------------------
  //  Walk quoted strings and pair them, REJECTING anything that is not a pair.
  //
  //  The first version took every two consecutive quoted strings as a pair. The
  //  file is an object whose VALUES are arrays of pairs:
  //
  //      {
  //        "oos_val": [
  //            [ "set a warning ...", "oos" ],
  //            [ "a show on broadway", "oos" ],
  //
  //  so the very first quoted string is the KEY "oos_val", not a text. Taking two at
  //  a time therefore paired ("oos_val", "set a warning ...") and then shifted every
  //  subsequent pair by one: each sentence became a LABEL and each label became a
  //  text.
  //
  //  Measured consequence: 23703 pairs were found -- the right COUNT -- with 19748
  //  distinct "labels" instead of the expected 151. A plausible total made the
  //  error invisible until a second statistic contradicted it.
  //
  //  The structural fix: in this shape a genuine pair has a comma between its two
  //  strings and a closing bracket after the second, while a key has a colon before
  //  the next string. Requiring both conditions rejects keys exactly, without
  //  needing a real JSON parser.
  //
  //  Stated as a rule: a parser that accepts anything shaped roughly right will
  //  produce a plausible count from wrong data, and the count is the hardest kind of
  //  evidence to doubt. Verify against a second, independent statistic.
  // ---------------------------------------------------------------------------
  auto next_nonspace = [&](size_t i) -> size_t {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    return i;
  };

  size_t i = 0;
  while (i < s.size()) {
    const size_t a = s.find('"', i);
    if (a == std::string::npos) break;
    size_t b = a + 1;
    std::string first;
    while (b < s.size() && s[b] != '"') {
      if (s[b] == '\\' && b + 1 < s.size()) { first.push_back(s[b+1]); b += 2; continue; }
      first.push_back(s[b]); ++b;
    }
    if (b >= s.size()) break;

    // between the two strings there must be a comma, not a colon
    const size_t sep = next_nonspace(b + 1);
    if (sep >= s.size() || s[sep] != ',') { i = b + 1; continue; }

    const size_t c = s.find('"', sep + 1);
    if (c == std::string::npos) break;
    size_t d = c + 1;
    std::string second;
    while (d < s.size() && s[d] != '"') {
      if (s[d] == '\\' && d + 1 < s.size()) { second.push_back(s[d+1]); d += 2; continue; }
      second.push_back(s[d]); ++d;
    }
    if (d >= s.size()) break;

    // and after the second there must be a closing bracket
    const size_t close = next_nonspace(d + 1);
    if (close >= s.size() || s[close] != ']') { i = d + 1; continue; }

    if (!first.empty() && !second.empty()) out.push_back({first, second});
    i = close + 1;
  }
  return out;
}

} // namespace canticle

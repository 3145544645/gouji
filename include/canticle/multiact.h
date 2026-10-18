// ===========================================================================
//  Multi-action request detection.
//
//  THE MEASURED PROBLEM
//  --------------------
//  The corpus contains ZERO multi-action examples, so the system has never seen
//  a request that asks for two things. What it does with one is worse than
//  failing, because it produces a well-formed call that mixes the two halves:
//
//      "read the file X and list the directory Y"  ->  list_dir {"path":"X"}
//      "what is 12 * 7 then list the directory Y"  ->  list_dir {"path":"12*7"}
//      "first read X then run the command hostname" ->  read_file {"path":"hostname"}
//
//  In each case the TOOL comes from the second clause and the PAYLOAD from the
//  first. Every one of those is syntactically valid JSON for a real tool, executes
//  without error, and does the wrong thing. A user reading the tool call would
//  have to notice the mismatch themselves.
//
//  WHY DETECT RATHER THAN ANSWER
//  -----------------------------
//  Teaching the model to emit two tool calls would require training data that does
//  not exist and a protocol change (the harness executes one call per turn). What
//  CAN be done reliably with no new data is recognise the pattern -- a conjunction
//  followed by a second verb phrase -- and say so.
//
//  Refusing with an explanation is strictly better than a silent wrong answer:
//  the user learns the limit immediately instead of after acting on bad output.
//  This is the same principle as the bare-path default: prefer the recoverable
//  outcome over the confident wrong one.
// ===========================================================================
#include <string>
#include <vector>
#include <cctype>
#include <cstring>

namespace canticle {

// A conjunction that joins two REQUESTS (not two objects).
//
// "and" is the ambiguous one: "read A and B" lists two files, "read A and list B"
// asks for two actions. The discriminator used here is the SECOND clause
// containing an action word, because that is what makes it a second request
// rather than a second argument.
// "GO AHEAD AND", "FEEL FREE TO", AND THE OTHER POLITE FILLERS.
//
// The conjunction test looks for " and " and then asks whether the tail contains
// an action word. "go ahead and run netstat" passes both halves -- there is an
// " and " and "run " follows it -- so it was refused as a two-action request.
//
// It is one request. "go ahead" is a politeness marker, not an action; the
// request is entirely in the second clause. The same holds for "feel free to",
// "be so kind as to", "if you would". These are fixed phrases, and the " and "
// inside one does not join two requests.
//
// Why a phrase list and not a grammar: the system has no parser, and the honest
// alternative to a list here is to accept the false refusal. A false refusal is
// recoverable (the user rephrases); a false ACTION on half a two-part request is
// not. So this list is allowed to grow when a measured case demands it, and the
// default stays "count it as two".
inline bool is_politeness_filler(const std::string& t, size_t conj_at) {
  // The text before the conjunction, trimmed, ends with one of these.
  static const char* kFill[] = {
    "go ahead", "go on", "feel free", "be so kind as", "if you would",
    "if you could", "when you get a chance", "when you can",
    "please go ahead", "kindly go ahead",
    // Added after measuring "do me a favor and run ping" as a two-action request:
    // the " and " is inside the fixed phrase, and only "run ping" is the request.
    "do me a favor", "do us a favor", "if you don't mind", "if you do not mind",
    "would you be so kind as", "be a dear and",
  };
  size_t e = conj_at;
  while (e > 0 && (t[e - 1] == ' ' || t[e - 1] == '\t')) --e;
  const std::string head = t.substr(0, e);
  for (const char* f : kFill) {
    const size_t n = std::strlen(f);
    if (head.size() < n) continue;
    if (head.compare(head.size() - n, n, f) == 0) return true;
  }
  return false;
}
inline bool is_conjunction(const std::string& t, size_t at, size_t* after) {
  static const char* kConj[] = {
    " then ", " and then ", " and also ", " and finally ", " after that ",
    " also ", " and ", "\xe5\xb9\xb6\xe4\xb8\x94", "\xe7\x84\xb6\xe5\x90\x8e", "\xe5\x86\x8d",
  };
  for (const char* c : kConj) {
    const size_t n = std::strlen(c);
    if (at + n <= t.size() && t.compare(at, n, c) == 0) { *after = at + n; return true; }
  }
  return false;
}

// Action words that mark a clause as a REQUEST rather than an argument.
//
// THE LIST WILL ALWAYS BE INCOMPLETE, and that is the lesson from defect 66.
// Measured miss:
//
//   "kindly show the contents of C:/log/log.json and could you work out 86 + 63"
//
// produced list_dir {"path":"86+63"} -- the tool from the first clause with the
// payload from the second, i.e. defect 33's mixing behaviour surviving inside a
// construction this list did not recognise. "work out" was absent, so the second
// clause did not read as a request.
//
// Adding "work out" fixes this instance and not the class, so two STRUCTURAL tests
// are added below: an arithmetic operand pair, and a polite prefix. Those cover the
// space instead of enumerating it.
inline bool has_action_word(const std::string& s) {
  static const char* kVerb[] = {
    // reads
    "read ", "list ", "write ", "save ", "show ", "display ", "open ", "print ",
    "view ", "fetch ", "cat ", "ls ", "dir ", "inspect ", "examine ", "peek ",
    // execution
    "run ", "execute ", "invoke ", "launch ", "call ", "start ",
    // arithmetic -- the words, not only the symbols
    "calculate ", "compute ", "what is ", "how much ", "evaluate ", "solve ",
    "work out ", "add ", "subtract ", "multiply ", "divide ", "sum ", "total ",
    // enumeration and search
    "enumerate ", "search ", "find ", "locate ", "grep ",
    // metadata
    "how big ", "how large ", "size of ", "when was ", "modified ",
    // creation and copying
    "make a ", "create a ", "mkdir ", "copy ", "duplicate ",
    // Chinese verbs, which the English-only list missed entirely.
    // Measured: "看看 D:/" could not be recognised as a first clause, and
    // "调用 hostname" could not be recognised as a second request, so both fell
    // through to whatever the surrounding words suggested.
    "\u8bf7\u6267\u884c ", "\u8bf7\u8fd0\u884c ", "\u6267\u884c ", "\u8fd0\u884c ",
    "\u8c03\u7528 ", "\u8bfb\u53d6 ", "\u67e5\u770b ", "\u6253\u5f00 ", "\u6253\u5370 ",
    "\u5217\u51fa ", "\u5217\u4e3e ", "\u770b\u770b ", "\u67e5\u627e ", "\u641c\u7d22 ",
    "\u8ba1\u7b97 ", "\u6c42 ", "\u5199\u5165 ", "\u4fdd\u5b58 ", "\u521b\u5efa ",
    "\u521b\u5efa\u4e00\u4e2a ", "\u65b0\u5efa ", "\u590d\u5236 ", "\u79fb\u52a8 ",
    "\u591a\u5927 ", "\u4fee\u6539\u65f6\u95f4 ",
  };
  for (const char* v : kVerb) {
    size_t p = 0;
    while ((p = s.find(v, p)) != std::string::npos) {
      const bool left_ok = (p == 0) || !std::isalnum((unsigned char)s[p - 1]);
      if (left_ok) return true;
      p += 1;
    }
  }

  // ---- structural test 1: an arithmetic operand pair ----
  // Digits around an operator have no other reading in this system, so the clause
  // is a request whether or not a known verb precedes it.
  for (size_t i = 1; i + 1 < s.size(); ++i) {
    const char c = s[i];
    if (c != '+' && c != '-' && c != '*' && c != '/') continue;
    if (std::isdigit((unsigned char)s[i-1]) && std::isdigit((unsigned char)s[i+1]))
      return true;
  }

  // ---- structural test 2: a polite prefix introduces a new request ----
  // By construction, whatever follows "could you" is a request, so the verb list
  // does not have to contain it.
  static const char* kPolite[] = {
    "could you ", "can you ", "would you ", "will you ", "please ",
    "let me ", "i want ", "i need ",
  };
  for (const char* v : kPolite) {
    size_t p = 0;
    while ((p = s.find(v, p)) != std::string::npos) {
      const bool left_ok = (p == 0) || !std::isalnum((unsigned char)s[p - 1]);
      if (left_ok) return true;
      p += 1;
    }
  }
  return false;
}

// True when a clause looks like a BARE COMMAND NAME.
//
// "hostname and whoami" asks for two commands but neither clause contains a verb:
// both halves ARE the command. A verb-only test misses it, and the system then
// answered with run_shell {"cmd":"whoami"} -- the second command, silently
// dropping the first.
//
// A bare command name is recognised by SHAPE rather than meaning, because there is
// no list of every executable: a single lower-case token, no spaces, no path
// separator, no digits, and at least one vowel-like letter so that an arbitrary
// fragment is not mistaken for a program.
inline bool is_bare_command(const std::string& clause) {
  std::string t = clause;
  while (!t.empty() && (t.back() == ' ' || t.back() == '\t')) t.pop_back();
  size_t s = t.find_first_not_of(" \t");
  if (s == std::string::npos) return false;
  t = t.substr(s);
  if (t.empty() || t.size() > 24) return false;
  if (t.find(' ') != std::string::npos) return false;
  if (t.find('/') != std::string::npos || t.find('.') != std::string::npos) return false;
  int letters = 0;
  for (char c : t) {
    if (std::isdigit((unsigned char)c)) return false;
    if (std::isalpha((unsigned char)c)) letters++;
    else if (c != '-' && c != '_') return false;
  }
  return letters >= 4;   // "ls" alone is too short to be sure; "whoami" is not
}

// True when the text asks for more than one action.
//
// Returns the offset of the conjunction that split it, so the caller can show the
// user exactly where the request was read as two.
inline bool detect_multi_action(const std::string& text, size_t* split_at = nullptr) {
  size_t at = 0;
  while (at < text.size()) {
    size_t after = 0;
    if (is_conjunction(text, at, &after)) {
      // A politeness filler before the conjunction means the " and " is part of a
      // fixed phrase ("go ahead and run X"), not a join between two requests.
      if (is_politeness_filler(text, at)) { ++at; continue; }
      const std::string tail = text.substr(after);
      if (has_action_word(tail) || is_bare_command(tail)) {
        if (split_at) *split_at = at;
        return true;
      }
    }
    ++at;
  }
  return false;
}

} // namespace canticle

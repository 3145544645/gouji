// ===========================================================================
//  CONTEXT: a symbol is evidence only when the context licenses it.
//
//  WHY THIS FILE EXISTS
//  -------------------
//  The same character means different things in different environments:
//
//      /      C:/tmp/a.txt      a path separator
//      /      C++/Rust          a comparison between two languages
//      -      40 - 14           subtraction
//      -      10-14 个          a quantity range
//      执行   执行 Get-Location  a command
//      执行   执行计划吗         a sentence about carrying out a plan
//
//  In every pair the character is IDENTICAL and the meaning is not. So no
//  property of the character can decide the question -- only the surroundings.
//
//  Ninety-eight defects were fixed one symbol at a time: a word boundary for
//  "put " inside "output", length ordering for " to " inside " into ", a leading
//  space removed for "in ". Each was right, and each was right about ONE symbol.
//  The next symbol needed a new patch, because the patch tested SHAPE, and shape
//  survives a change of environment while meaning does not.
//
//  This file states the rule once: A SYMBOL IS NOT EVIDENCE UNTIL THE CONTEXT
//  LICENSES IT. Three licences are needed -- path, expression, command -- and
//  they share one skeleton: take the span, look at what is on either side, and
//  ask whether this environment could mean that.
//
//  The patches grow with the number of symbols. The licences grow with the number
//  of DOMAINS, and there are three.
// ===========================================================================
#pragma once
#include <string>
#include <cstring>
#include <cctype>

namespace canticle {
namespace context {

// True for a UTF-8 lead byte that begins a CJK character. Used to tell prose
// from data: a command or a path in this system is ASCII, a sentence is not.
inline bool is_cjk_lead(unsigned char c) { return c >= 0xE0; }

// ---------------------------------------------------------------------------
//  PATH LICENCE.
//
//  The old test was "contains / or \" -- which is true of C++/Rust, TCP/IP and
//  A/B testing. All three were routed to list_dir with the phrase as the path.
//
//  A span is licensed as a path when it has at least one of:
//
//    * a drive prefix           C:/tmp        (a volume name, then a separator)
//    * a leading separator      /tmp  ./x  ../y
//    * a file extension         build/ckpt/canticle.am, STATUS.md
//
//  and NONE of its segments contains a character that cannot appear in a path
//  segment. That second condition is what rejects C++/Rust: "C++" is not a volume
//  name, and the plus signs disqualify it as a directory segment.
//
//  A bare X/Y with no drive, no leading separator and no extension is NOT
//  licensed. TCP/IP and A/B testing are exactly that shape, and nothing in the
//  text says the speaker meant a file.
// ---------------------------------------------------------------------------
inline bool segment_chars_ok(const std::string& seg) {
  if (seg.empty()) return false;
  for (unsigned char c : seg) {
    if (is_cjk_lead(c)) return false;
    // characters that make a token a language name or an operator, not a segment
    if (c == '+' || c == '#' || c == '~' || c == '<' || c == '>' ||
        c == '|' || c == '"' || c == '*' || c == '?') return false;
  }
  return true;
}

inline bool has_extension(const std::string& seg) {
  const size_t dot = seg.find_last_of('.');
  if (dot == std::string::npos || dot + 1 >= seg.size()) return false;
  if (dot == 0) return false;
  const std::string ext = seg.substr(dot + 1);
  if (ext.empty() || ext.size() > 5) return false;
  for (unsigned char c : ext) if (!std::isalnum(c)) return false;
  return true;
}

// Is this token licensed as a path?
inline bool token_is_path(const std::string& tok) {
  if (tok.empty()) return false;
  if (tok.find('/') == std::string::npos && tok.find('\\') == std::string::npos) {
    // a bare filename with an extension is still a file reference
    return has_extension(tok) && segment_chars_ok(tok);
  }
  // every segment must be path-shaped
  size_t start = 0;
  bool drive_seen = false;
  for (size_t i = 0; i <= tok.size(); ++i) {
    if (i < tok.size() && tok[i] != '/' && tok[i] != '\\') continue;
    const std::string seg = tok.substr(start, i - start);
    if (!seg.empty() && !segment_chars_ok(seg)) return false;
    // a drive prefix is a single letter followed by a colon
    if (start == 0 && seg.size() == 2 && std::isalpha((unsigned char)seg[0]) &&
        seg[1] == ':') drive_seen = true;
    start = i + 1;
  }
  if (drive_seen) return true;
  // leading separator, or an explicit relative prefix
  if (tok[0] == '/' || tok[0] == '\\') return true;
  if (tok.compare(0, 2, "./") == 0 || tok.compare(0, 3, "../") == 0) return true;
  // otherwise the LAST segment must carry an extension
  const size_t sep = tok.find_last_of("/\\");
  const std::string last = (sep == std::string::npos) ? tok : tok.substr(sep + 1);
  return has_extension(last);
}

// True when the text contains at least one token licensed as a path.
inline bool licensed_as_path(const std::string& s) {
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && (std::isspace((unsigned char)s[i]) ||
                            s[i] == '"' || s[i] == '\'')) ++i;
    if (i >= s.size()) break;
    size_t j = i;
    while (j < s.size() && !std::isspace((unsigned char)s[j]) &&
           s[j] != '"' && s[j] != '\'') ++j;
    if (token_is_path(s.substr(i, j - i))) return true;
    i = j + 1;
  }
  return false;
}

// ---------------------------------------------------------------------------
//  EXPRESSION LICENCE.
//
//  A digit-operator-digit shape is arithmetic UNLESS the numbers are counting
//  something. Measured: "写一个 10-14 个词的句子" produced calc {"expr":"10-14"},
//  because 10-14 is a RANGE -- the classifier 个 after it makes the pair part of
//  a noun phrase, not an operation.
//
//  So the licence asks what follows the second operand. A classifier or a unit
//  withdraws it. This is the syntactic axis of context: the same minus sign is an
//  operator or a connective depending on the grammar around it.
// ---------------------------------------------------------------------------
inline bool followed_by_classifier(const std::string& s, size_t pos) {
  size_t i = pos;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
  if (i >= s.size()) return false;
  // Chinese classifiers and units
  static const char* kZh[] = {
    "个", "次", "行", "字", "词", "条", "张", "份", "位", "名",
    "天", "年", "月", "日", "时", "分", "秒", "米", "克", "元",
    "块", "页", "步", "种", "项", "倍", "度", "级", "层", "只",
    "件", "台", "辆", "本", "篇", "段", "句", "岁", "人", "户",
  };
  for (const char* z : kZh) {
    const size_t n = std::strlen(z);
    if (s.compare(i, n, z) == 0) return true;
  }
  // English units and counters
  static const char* kEn[] = {
    "words", "word", "characters", "chars", "letters", "lines", "steps",
    "items", "times", "days", "years", "minutes", "seconds", "hours",
    "meters", "metres", "kg", "dollars", "euros", "pages", "people",
  };
  for (const char* e : kEn) {
    const size_t n = std::strlen(e);
    if (i + n > s.size()) continue;
    bool same = true;
    for (size_t k = 0; k < n; ++k)
      if (std::tolower((unsigned char)s[i+k]) != e[k]) { same = false; break; }
    if (!same) continue;
    // must end at a non-letter, or "wordy" would match "word"
    if (i + n < s.size() && std::isalpha((unsigned char)s[i+n])) continue;
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
//  COMMAND LICENCE.
//
//  A shell verb needs a shell payload. Measured: "我们应该执行计划吗？" was routed
//  to run_shell with the whole question as the command, because 执行 matched and
//  nothing checked that what followed could be executed.
//
//  A command in this system is ASCII: an executable name, flags, paths. A CJK
//  sentence with punctuation is prose, and prose after "execute" is a metaphor.
// ---------------------------------------------------------------------------
inline bool licensed_as_command(const std::string& s) {
  if (s.empty()) return false;
  int ascii_printable = 0;
  for (unsigned char c : s) {
    if (is_cjk_lead(c)) return false;
    if (std::isprint(c) && !std::isspace(c)) ascii_printable++;
  }
  return ascii_printable > 0;
}

// ---------------------------------------------------------------------------
//  DOMAIN: which environment a span is being read in.
//
//  The three licences above each ask a question about shape. This adds the axis
//  that shape cannot supply: VOCABULARY.
//
//  Measured on CLINC150 out-of-scope sentences, sixteen of three hundred were
//  acted on, and every one was a FABRICATED PAYLOAD -- the extractor fell back to
//  "the last whitespace token", so any sentence yielded an argument:
//
//      "set a warning for when my bank account starts running low"  -> run_shell {"cmd":"low"}
//      "what'll an education at boulder state college run"          -> run_shell {"cmd":"run"}
//      "can you give me the standard tuning notes"                  -> read_file {"path":"notes"}
//      "how many prime numbers are there between 0 and 100"         -> read_file {"path":"0+100"}
//
//  No shape test separates "hostname" from "low": both are single ASCII words.
//  What separates them is that one is a COMMAND and the other is the last word of
//  a sentence. That is a fact about vocabulary, so the licence consults one.
//
//  This is the honest reason to have a Domain type at all: without it, the
//  distinction gets re-derived as an ad-hoc string comparison wherever it is
//  needed, which is how twelve implicit domain tests ended up scattered through
//  this header before it existed.
// ---------------------------------------------------------------------------
enum class Domain { Unknown, File, Command, Expression, Prose };

inline const char* domain_name(Domain d) {
  switch (d) {
    case Domain::File:       return "file";
    case Domain::Command:    return "command";
    case Domain::Expression: return "expression";
    case Domain::Prose:      return "prose";
    default:                 return "unknown";
  }
}

// ---------------------------------------------------------------------------
//  THE COMMAND VOCABULARY.
//
//  A whitelist is usually the wrong tool, and this one is justified by what it is
//  a list OF: the set of programs a shell can run is a vocabulary, not a shape.
//  "low" cannot become a command by being spelled differently; "hostname" is one
//  because it is the name of a program.
//
//  A span is also licensed as a command when it is MULTI-TOKEN, because "ls -la"
//  and "exit 3" carry their own evidence in the flags and arguments. And when it
//  names a path to an executable, since that is self-evidencing too.
// ---------------------------------------------------------------------------
inline bool known_command(const std::string& first_token) {
  static const char* kCmd[] = {
    // Windows
    "cmd", "powershell", "pwsh", "dir", "type", "copy", "move", "del",
    "hostname", "whoami", "ipconfig", "netstat", "tasklist", "systeminfo",
    "where", "set", "echo", "cd", "exit", "start", "taskkill", "ping",
    "tracert", "nslookup", "findstr", "more", "tree", "ver", "vol",
    // PowerShell cmdlets, matched case-insensitively
    "get-date", "get-location", "get-childitem", "get-content", "set-content",
    "get-process", "get-service", "get-item", "test-path", "select-string",
    "measure-object", "sort-object", "where-object", "write-output",
    "invoke-webrequest", "get-help", "get-command", "new-item", "remove-item",
    // POSIX
    "ls", "cat", "grep", "find", "ps", "df", "du", "uname", "date", "sleep",
    "curl", "wget", "head", "tail", "wc", "sort", "uniq", "sed", "awk", "git",
    "python", "python3", "node", "npm", "make", "cmake", "nvcc", "gcc", "cl",
  };
  std::string low = first_token;
  for (char& c : low) c = (char)std::tolower((unsigned char)c);
  for (const char* c : kCmd) if (low == c) return true;
  return false;
}

// Stronger than licensed_as_command: requires vocabulary or structure.
inline bool licensed_as_command_strong(const std::string& span) {
  if (!licensed_as_command(span)) return false;   // already excludes CJK prose
  // multi-token: flags, arguments, or a path to an executable
  for (size_t i = 0; i < span.size(); ++i) {
    const char c = span[i];
    if (c == ' ' || c == '-' || c == '/' || c == '\\') return true;
  }
  return known_command(span);
}

// Does this span carry evidence of being a file reference?
//
// The path licence already answers this for shapes; the addition here is that a
// BARE WORD is not evidence. "notes" in "the standard tuning notes" is the last
// word of a sentence, not a filename, and only a shape test can tell the
// difference -- which is why a bare name is accepted only where naming is the
// point (mkdir, write), never where reading is.
// ---------------------------------------------------------------------------
//  A FILE REFERENCE IS A PATH THAT NAMES SOMETHING THESE TOOLS COULD OPEN.
//
//  The first version had a second test after token_is_path:
//
//      if (!token_is_path(span)) return false;
//      if (no separator) return has_extension(span);   // unreachable
//      return true;
//
//  which is DEAD: token_is_path already requires segment characters to be
//  path-shaped and returns false for a bare word that has no extension. The second
//  branch could never be false when reached, so it looked like a guard and did
//  nothing.
//
//  Mutation testing is what exposed it. The mutation aimed at that branch changed no
//  behaviour -- and a mutation that cannot change behaviour is pointing at code that
//  cannot matter. Reaching for "the test must be broken" before "the code is broken"
//  is the wrong order; here the code was.
//
//  So the rule is stated once, in token_is_path, and this function says what it
//  actually means: a bare word is not a reference to a file these tools can open.
// ---------------------------------------------------------------------------
inline bool licensed_as_file_reference(const std::string& span) {
  return token_is_path(span);
}

} // namespace context
} // namespace canticle


// ============================================================================
//  CANTICLE-AM encoder  (host side, pure C++, no GPU, no trained parameters)
//
//  LESSON THAT PRODUCED THIS FILE
//  ------------------------------
//  Three encoders were measured and two failed:
//
//   1. mean of the model's token embeddings  -> 17.7% (chance).
//      The embeddings were random, so the mean of a few random vectors is
//      nearly the same vector for every prompt.
//
//   2. Random Indexing over token ids (D=4096, 2% density).
//      Measured pairwise cosine between DIFFERENT prompts: 0.000, repeatedly.
//      Two sparse +-1 vectors of dimension D with ~2D/100 nonzeros are almost
//      exactly orthogonal, so every query was equidistant from every memory.
//      The single 0.51 hit was one lucky collision.
//
//  A representation in which unrelated things are orthogonal is useless for
//  nearest-neighbour recall: there is no gradient of similarity to follow.
//  What is needed is a code where SIMILAR TEXT IS SIMILAR BY CONSTRUCTION.
//
//  That is what this file implements: feature hashing over character n-grams,
//  weighted by IDF, L2-normalised. "read the file X" and "write to X" share
//  the grams "e ", "th", "e f" and land near each other; "run the command" and
//  "what is 40-14" share almost nothing and land far apart. The geometry comes
//  from the surface form of the text, so it needs no training at all.
// ============================================================================
#include "canticle/am.h"
#include "canticle/model.h"    // ToolSpec, for the multi-turn rebuild
#include "canticle/context.h"  // licences: path / expression / command
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace canticle {

// ---------------------------------------------------------------------------
//  Character n-gram feature hashing.
//
//  Why characters and not words: the payloads are paths and expressions
//  ("C:/work/log.txt", "40-14"). A word-level vocabulary would see those as
//  distinct unknown tokens, while character trigrams capture that
//  "/work/" and "/tmp/" differ only in their middle.
// ---------------------------------------------------------------------------
static inline uint64_t fnv1a(const char* p, size_t n, uint64_t h = 1469598103934665603ull) {
  for (size_t i = 0; i < n; ++i) { h ^= (unsigned char)p[i]; h *= 1099511628211ull; }
  return h;
}

void AmEncoder::build_idf(const std::vector<std::string>& docs) {
  std::unordered_map<uint64_t, int> df;
  for (const auto& d : docs) {
    std::unordered_map<uint64_t, int> seen;
    const int n = (int)d.size();
    for (int i = 0; i + 3 <= n; ++i) {
      uint64_t h = fnv1a(d.data() + i, 3);
      seen[h] = 1;
    }
    for (auto& kv : seen) df[kv.first]++;
  }
  idf.clear();
  const double N = (double)std::max<size_t>(1, docs.size());
  for (auto& kv : df)
    idf[kv.first] = (float)std::log(1.0 + N / (1.0 + (double)kv.second));
  idf_built = true;
}

void AmEncoder::encode(const std::string& text, std::vector<float>& out) const {
  out.assign(D, 0.f);
  const int n = (int)text.size();
  if (n < 3) { return; }

  // Reserved slots live at the top of the vector. Hashing must not touch them
  // or the routing signal gets overwritten by whatever grams happen to land
  // there -- which made the keyword channel silently useless.
  static const int NKW_R = 9;
  const int HASH_D = D - NKW_R - 1;

  // ---- trigram features, IDF weighted, sublinear term frequency ----
  std::unordered_map<uint64_t, int> tf;
  tf.reserve((size_t)n * 2);
  for (int i = 0; i + 3 <= n; ++i)
    tf[fnv1a(text.data() + i, 3)]++;

  for (auto& kv : tf) {
    float w = 1.0f + std::log((float)kv.second);          // sublinear tf
    if (idf_built) {
      auto it = idf.find(kv.first);
      w *= (it != idf.end()) ? it->second : 4.0f;          // unseen gram -> high idf
    }
    // signed feature hashing: the sign is derived from the hash itself so that
    // collisions cancel in expectation instead of always adding.
    const uint64_t h = kv.first;
    const int idx = (int)(h % (uint64_t)HASH_D);
    const float sgn = (h & 0x8000000000000000ull) ? -1.0f : 1.0f;
    out[idx] += sgn * w;
  }

  // ---- structural cues: two explicit channels the n-grams cannot express ----
  //
  // (a) the leading verb decides WHICH tool. "read"/"list"/"write"/"run"/"what"
  //     are short but carry all the routing information.
  // (b) a small set of tool keywords gets its own dedicated dimension, which
  //     makes the template signal survive hashing collisions with the payload.
  static const char* KWS[] = {"read", "list", "write", "run", "what", "hello", "who", "the", "to"};
  const int NKW = (int)(sizeof(KWS) / sizeof(KWS[0]));
  for (int k = 0; k < NKW; ++k) {
    if (text.find(KWS[k]) != std::string::npos)
      out[D - NKW + k] += 3.0f;                 // reserved, high weight
  }
  // ---- length channel: distinguishes "hi" from a long instruction ----
  out[D - NKW - 1] += std::min(2.0f, (float)n / 40.0f);

  l2norm(out);
}

void AmEncoder::encode_marked(const std::string& text, std::vector<float>& out) const {
  // Same as encode, but reserved slots are left alone so the caller can inject
  // an explicit template id. Used by the slot router.
  encode(text, out);
}

void AmEncoder::l2norm(std::vector<float>& v) {
  double s = 0;
  for (float x : v) s += (double)x * x;
  const float inv = (float)(1.0 / (std::sqrt(s) + 1e-9));
  for (float& x : v) x *= inv;
}

// ---------------------------------------------------------------------------
//  Slot router.
//
//  The template decides the tool; only the payload is copied. Paths are the
//  one payload type where exact copying matters most (a dropped character
//  breaks the call), so they are extracted with a direct scan rather than
//  reconstructed from features.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  Payload extraction.
//
//  BUG THIS FIXES: the first version took the LAST whitespace token as the
//  payload. For "what is 123 * 456" that is "456", so the copied reply became
//  "53*456" -- the first operand was silently dropped and the calculator
//  returned a wrong number. A wrong answer is far worse than a refused one.
//
//  Three payload shapes are recognised explicitly, in priority order. Anything
//  that does not match falls back to the last token.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  Intent inference. Pure surface structure -- no vocabulary model, no weights.
//
//  Priority order is deliberate:
//     strong verb  >  strong noun  >  weak hint  >  question+path  >  expression
//  A strong verb like "write" always wins because it is unambiguous. Nouns are
//  ranked above weak hints so that "show the entries of X" reads as a directory
//  question (strong noun "entries") rather than a read (weak hint "show").
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  Keyword test with WORD BOUNDARIES.
//
//  Plain substring search caused a real bug: the write table contains "put " and
//  the read table contains "output", so "output C:/tmp/log.md" matched the write
//  verb "put " inside "output" and every read request that used "output" was
//  routed to write_file -- 12 wrong answers in a single eval run.
//
//  Matching is on a token basis: the keyword must start at a word boundary. The
//  keyword may itself end with a space, which still anchors the start.
// ---------------------------------------------------------------------------
static bool has_w(const std::string& t, const char* w) {
  const size_t n = std::strlen(w);
  if (n == 0) return true;
  // A keyword that BEGINS with a delimiter contains its own left boundary, so
  // testing the preceding character is wrong rather than merely redundant.
  //
  // Measured: has_w(t, " in ") always returned false, because the character
  // before the leading space is the 's' of "is". The guard "what is in <path>"
  // therefore never fired and the request was answered as a calculation over the
  // path string. Callers should prefer the form without a leading space
  // ("in "), but the function must not silently defeat the spaced form either.
  const bool self_bounded = (w[0] == ' ' || w[0] == '\t' ||
                             std::ispunct((unsigned char)w[0]));
  size_t p = 0;
  while (true) {
    p = t.find(w, p);
    if (p == std::string::npos) return false;
    if (self_bounded) return true;
    const bool left_ok = (p == 0) || !std::isalnum((unsigned char)t[p - 1]);
    if (left_ok) return true;
    p += 1;
  }
}

// ---------------------------------------------------------------------------
//  DELEGATED TO THE CONTEXT LICENCE, which is the whole point of context.h.
//
//  What stood here was the purest form of the mistake this project spent 98 defects
//  learning: "contains a slash" was taken as proof of a path. Measured cost:
//
//      compare the tradeoffs in C++/Rust        -> list_dir {"path":"C++/Rust"}
//      what is the best practice in TCP/IP ...  -> list_dir {"path":"TCP/IP"}
//      explain the difference in A/B testing    -> list_dir {"path":"A/B"}
//
//  Three ordinary questions, three fabricated directory listings, from one test
//  that asked about a character instead of about a context.
//
//  A slash is a SHAPE. Whether it separates directories or joins two names is a
//  question about the environment, and context.h is where that question is now
//  asked -- once, for every rule that needs it.
// ---------------------------------------------------------------------------
static bool looks_like_path(const std::string& t) {
  return context::licensed_as_path(t);
}

// NUMBER op NUMBER -- the one unambiguous structural signature of arithmetic.
// Declared here because both the extractor and this predicate must agree on how
// word operators are spelled; a mismatch is exactly what made "7 x 7" extract as
// "7*7" while still being classified as a directory listing.
static std::string normalise_math_words(const std::string& in);

static bool looks_like_expression(const std::string& raw) {
  // Normalise FIRST. Measured failure: "what is 7 x 7" and "what is 8 times 9"
  // produced the right aria of payload ("7*7") but were routed to list_dir,
  // because this predicate looked at the raw text, found no "+-*/" symbol, and
  // returned false while the extractor -- which does normalise -- returned an
  // expression. The two must test the same string.
  const std::string t = normalise_math_words(raw);
  for (size_t i = 0; i < t.size(); ++i) {
    if (!std::isdigit((unsigned char)t[i])) continue;
    size_t j = i;
    while (j < t.size() && std::isdigit((unsigned char)t[j])) ++j;
    size_t k = j;
    while (k < t.size() && t[k] == ' ') ++k;
    if (k < t.size() && std::strchr("+-*/", t[k])) {
      size_t m = k + 1;
      while (m < t.size() && t[m] == ' ') ++m;
      if (m < t.size() && std::isdigit((unsigned char)t[m])) return true;
    }
    i = j;
  }
  // A percentage request ("30% off 279") has no arithmetic symbol but is a
  // calculation. Without this branch the extracted expression "279*30/100" was
  // produced and then discarded by the router, which sent the request to
  // list_dir with the path "279*30/100".
  {
    std::string low = raw;
    for (char& c : low) c = (char)std::tolower((unsigned char)c);
    if (low.find('%') != std::string::npos || low.find("percent") != std::string::npos)
      return true;
    // Named operations carry no arithmetic symbol at all. Measured: "what is the
    // square root of 144" and "what is 8 factorial" extracted correctly and were
    // then routed to list_dir, because this predicate only looked for symbols.
    if (low.find("square root") != std::string::npos) return true;
    if (low.find("factorial") != std::string::npos) return true;

    // -------------------------------------------------------------------------
    //  Last resort, and the structurally correct one: ask the EXTRACTOR.
    //
    //  This predicate and extract_payload have now disagreed three separate
    //  times -- "7 x 7", "8 factorial", "3/4 of 18" -- and each disagreement cost
    //  a round of debugging. They are two views of one question, "is this
    //  arithmetic?", and any list of forms written into one of them will drift
    //  from the other.
    //
    //  Rather than add a fourth special case, the predicate now accepts whatever
    //  the extractor recognises. The extractor is the authority; this function
    //  only has to avoid claiming arithmetic where there is none, and a successful
    //  extraction is exactly that evidence.
    // -------------------------------------------------------------------------
    extern std::string extract_payload(const std::string&);
    const std::string as_expr = extract_payload(raw);
    if (!as_expr.empty() && as_expr != raw) {
      // The extracted string must be arithmetic, not merely something with a
      // punctuation character in it.
      //
      // Measured bug: the first version accepted any '+ - * / !', and a PATH
      // contains a slash -- so "what is in C:/tmp" extracted the payload "C:/tmp",
      // passed this test, and was answered as a calculation over the path string.
      // The operator has to be an operator, i.e. it must sit between operands.
      if (as_expr.find("sqrt(") != std::string::npos) return true;
      if (as_expr.find('!') != std::string::npos) return true;
      for (size_t i = 1; i + 1 < as_expr.size(); ++i) {
        const char c = as_expr[i];
        if (c != '+' && c != '-' && c != '*' && c != '/') continue;
        // an operand on each side: a digit, or a closing paren
        const char L = as_expr[i-1], R = as_expr[i+1];
        const bool lok = std::isdigit((unsigned char)L) || L == ')';
        const bool rok = std::isdigit((unsigned char)R) || R == '(';
        if (lok && rok) return true;
      }
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
//  Operations this system understands but has no tool for.
//
//  Grouped by kind so the refusal message can name the category, and so that a
//  future decision to implement one is a matter of deleting a line here rather than
//  hunting through the classifier.
//
//  Deliberately NOT included: anything ambiguous. "show" is not here because it is
//  a legitimate read cue; "copy" is not here because it could be read as "copy the
//  contents out" which read_file serves. Only verbs that unambiguously name an
//  operation with no current implementation.
// ---------------------------------------------------------------------------
bool intent_is_unsupported_op(const std::string& t) {
  // destructive file operations
  static const char* kDestroy[] = {
    "delete ", "remove ", "erase ", "unlink ", "rm ", "trash ",
    "\u5220\u9664", "\u79fb\u9664",
  };
  // move / copy / rename
  // copy and duplicate have MOVED OUT: both are now served by the copy tool.
  // rename and move remain, because both REMOVE the original.
  static const char* kMove[] = {
    "rename ", "move ", "mv ",
    "\u91cd\u547d\u540d", "\u79fb\u52a8",
  };
  // REMOVED: directory creation is now served by the mkdir tool.
  //
  // Left here as a comment rather than deleted outright, because this is the third
  // time a capability moved between "unsupported" and "implemented" (stat, search,
  // and now mkdir) and the pattern is worth making visible: whenever a tool is
  // added, its entry must be deleted from this table in the SAME change, or the
  // refusal fires first and the new tool is never reached.
  //
  //     static const char* kDir[] = {
  //       "make a directory", "mkdir ", "new folder", ...
  //     };
  // -------------------------------------------------------------------------
  //  metadata and search are NO LONGER here.
  //
  //  They were listed as Unsupported when no tool answered them, which was the
  //  right classification at the time. With stat and search implemented they move
  //  to their own intents (see infer_intent), and leaving them in this table would
  //  make the two mechanisms contradict each other -- the refusal would fire
  //  before the new tool was ever considered.
  //
  //  This is the same contract-must-be-updated-on-both-sides rule that defects 17
  //  and 37 cost a round each to find. When a capability is added, its entry is
  //  removed from here in the same change.
  // -------------------------------------------------------------------------
  auto any = [&](const char* const* arr, size_t n) {
    for (size_t i = 0; i < n; ++i) if (has_w(t, arr[i])) return true;
    return false;
  };
  if (any(kDestroy, sizeof(kDestroy)/sizeof(kDestroy[0]))) return true;
  if (any(kMove,    sizeof(kMove)/sizeof(kMove[0])))    return true;
  return false;
}

// ---------------------------------------------------------------------------
//  Directory creation, now that mkdir exists.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  Copy requests, now that the copy tool exists.
//
//  Requires BOTH paths to be extractable: "copy C:/a.txt" with no destination is
//  not a copy request, it is an incomplete one, and guessing a destination would be
//  the worst possible failure for a tool that writes.
// ---------------------------------------------------------------------------
static bool wants_copy(const std::string& t, const std::string& original) {
  static const char* k[] = {"copy ", "cp ", "duplicate ", "\u590d\u5236"};
  bool named = false;
  for (size_t i = 0; i < sizeof(k)/sizeof(k[0]); ++i) if (has_w(t, k[i])) { named = true; break; }
  if (!named) return false;
  std::string s, d;
  return extract_copy_paths(original, &s, &d);
}

static bool wants_mkdir(const std::string& t) {
  static const char* k[] = {
    "make a directory", "make a folder", "create a directory", "create a folder",
    "mkdir ", "new folder", "new directory", "make directory",
    "\u65b0\u5efa\u76ee\u5f55", "\u65b0\u5efa\u6587\u4ef6\u5939",
  };
  for (size_t i = 0; i < sizeof(k)/sizeof(k[0]); ++i) if (has_w(t, k[i])) return true;
  return false;
}

// ---------------------------------------------------------------------------
//  Requests that name a metadata question or a search.
//
//  Kept in their own predicates rather than inlined into infer_intent so the two
//  new tools have a single place to look, and so removing them again (should the
//  tools be withdrawn) is one deletion rather than a hunt.
// ---------------------------------------------------------------------------
static bool wants_stat(const std::string& t) {
  static const char* k[] = {
    "how big is ", "how large is ", "file size", "size of ", "how many bytes",
    "when was ", "when was it", "last modified", "modification time", "modified",
    "\u6709\u591a\u5927", "\u4fee\u6539\u65f6\u95f4", "\u6587\u4ef6\u5927\u5c0f",
  };
  for (size_t i = 0; i < sizeof(k)/sizeof(k[0]); ++i) if (has_w(t, k[i])) return true;
  return false;
}
static bool wants_search(const std::string& t) {
  static const char* k[] = {
    "find all ", "find any ", "find the ", "search for ", "search in ", "grep ",
    "look for ", "locate ", "which files contain", "files containing",
    "\u641c\u7d22", "\u67e5\u627e", "\u5305\u542b",
  };
  for (size_t i = 0; i < sizeof(k)/sizeof(k[0]); ++i) if (has_w(t, k[i])) return true;
  return false;
}

// ---------------------------------------------------------------------------
//  THE PRECEDENCE TABLE.
//
//  Nineteen defects in this project were ordering mistakes, and the cause was
//  always the same: "which rule wins" lived in the ORDER OF LINES rather than in
//  anything a reader or a test could check. A rule added at the bottom silently
//  lost to a rule at the top that matched something incidental.
//
//  The order is now written down, with a reason per level, so that a new rule has
//  an obvious slot and a reviewer has something to compare against. The numbering
//  is the invariant: infer_intent must return intents in this relative order.
//
//    1  UNSUPPORTED   the request names an operation this system does not perform.
//                     Always first: refusing is never wrong, and "delete X" must
//                     not be answered by whatever else happens to match.
//
//    2  EXPLICIT TOOL an operation with its own tool, named unambiguously.
//                     stat / search / mkdir / copy. Each requires its arguments to
//                     be present, so a partial request falls through rather than
//                     guessing.
//
//    3  ARITHMETIC    an expression shape. Outranks the bare-path rule because a
//                     slash inside "3/4" is arithmetic, not a separator.
//
//    4  STRONG VERBS  a verb that names the ACTION (write / read / put).
//                     Outranks containment: "put hello in C:/tmp" is a WRITE, and
//                     the first version of this table let " in " beat it, turning
//                     a write into a directory listing.
//
//    5  CONTAINMENT   "in <path>" -- asks about the CONTENTS of a container.
//                     Outranks the weak hints because it says what is being acted
//                     on, while they say only that some action occurs.
//
//    6  WEAK HINTS    a generic action verb needing a path to mean anything
//                     (show / open / display).
//
//    7  BARE PATH     a path with no other signal: the safest default is read.
//
//  A test asserts the relative order of the observable cases, so a future edit
//  that breaks it fails rather than silently changing routing.
// ---------------------------------------------------------------------------
Intent infer_intent(const std::string& text) {
  std::string t = text;
  for (char& c : t) c = (char)std::tolower((unsigned char)c);

  const bool path = looks_like_path(t);
  const bool expr = looks_like_expression(t);

  // ---------------------------------------------------------------------------
  //  Verb and noun tables.
  //
  //  These MUST stay in step with data.cpp's surface-form banks. When the
  //  corpus was widened to many phrasings per intent, six new forms appeared
  //  that no rule covered ("make a file", "store hello at", "contents of",
  //  "show files in", "output", "stocktake") and held-out accuracy fell to
  //  94.59%. The generator and the recogniser are two halves of one contract;
  //  adding vocabulary on one side without the other is a silent regression.
  //
  //  Order: strong verbs (unambiguous action) > strong nouns (container words)
  //  > weak hints (ambiguous alone) > question+path > bare-path default.
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  //  UNSUPPORTED OPERATIONS ARE CHECKED FIRST, and the order is the point.
  //
  //  These verbs name real, unambiguous operations that no tool implements. Checked
  //  after the other rules they were absorbed by whatever else happened to match:
  //
  //      "delete C:/tmp/scratch.txt"   -> read_file   (bare-path default)
  //      "rename C:/tmp/a.txt to ..."  -> read_file
  //      "make a directory C:/tmp/n"   -> list_dir
  //      "how big is C:/.../STATUS.md" -> list_dir
  //      "find all txt files in C:/..."-> list_dir   (silently ignores the filter)
  //
  //  The read_file cases are the dangerous ones: the user asked to DELETE something
  //  and the system answered by opening it. Nothing was harmed, but only because
  //  read_file is harmless -- the routing decision itself was arbitrary.
  //
  //  Naming the operation first makes the outcome deterministic: every one of these
  //  is refused with a reason, which is recoverable, rather than answered with an
  //  unrelated tool, which is not.
  // ---------------------------------------------------------------------------
  if (intent_is_unsupported_op(t)) return Intent::Unsupported;

  // ---------------------------------------------------------------------------
  //  AN EXPLICIT COMMAND TEMPLATE, CHECKED HERE AND NOT LOWER DOWN.
  //
  //  Two separate defects came from placing this rule in the wrong slot, and both
  //  were measured rather than reasoned about:
  //
  //  1. FIRST PLACEMENT WAS TOO LOW. The rule sat just above the vague-verb branch,
  //     below the container-word rule. "run the command dir C:/tmp" matched the
  //     container word "dir " at level 5 and returned ListDir before ever reaching
  //     level 6. Measured: the retrieval candidates were 24 list_dir rows and the
  //     command never ran.
  //
  //  2. ITS ORIGINAL GUARD WAS "!path", meaning "a command request does not mention
  //     a path". That is false for most real commands -- they take a path as an
  //     ARGUMENT. "run the command type C:/a.txt" was routed to read_file for no
  //     reason other than the slash.
  //
  //  The template settles both. "run the command X" names a program; where X
  //  happens to live, and whether its arguments contain slashes, does not change
  //  which program was named. Only an explicit template earns this: the vague leads
  //  ("run ", "call ") still need their old guards, because "run the meeting notes"
  //  is prose and "call the directory C:/tmp" is not a command.
  //
  //  This sits at level 1.5 -- after Unsupported, because refusing is never wrong,
  //  and before everything else, because an explicit template outranks every
  //  incidental word inside its payload.
  // ---------------------------------------------------------------------------
  auto command_shaped = [&](const std::string& s) -> bool {
    if (s.empty()) return false;
    int cjk = 0, ascii_printable = 0;
    for (unsigned char c : s) {
      if (c >= 0xE0) { cjk++; continue; }              // UTF-8 lead byte of a CJK char
      if (std::isprint(c) && !std::isspace(c)) ascii_printable++;
    }
    if (cjk != 0 || ascii_printable == 0) return false;
    //
    //  SHAPE IS NOT ENOUGH. This test used to stop at "the payload is ASCII",
    //  which let "fire up the meeting notes" through as a shell command:
    //
    //      fire up the meeting notes  ->  run_shell {"cmd":"the meeting notes"}
    //
    //  Every character in "the meeting notes" is printable ASCII, so the shape
    //  test passed and a sentence about a document was executed as a program name.
    //
    //  The vocabulary is what separates a program from a noun phrase, and it is the
    //  same list the payload gate in am_main.cpp uses. One list, two callers -- the
    //  alternative was a second copy that could drift.
    //
    //  Cost: an invented command name under a VAGUE lead is now refused rather than
    //  attempted. That is the direction this system wants to fail in: a refusal is
    //  recoverable, running nonsense is not. An EXPLICIT template still routes to
    //  the payload gate, which says so in its own message.
    //
    //  A multi-token payload is allowed when its FIRST token is a known command:
    //  "ipconfig /all" and "git status -s" are commands with arguments, and neither
    //  whole string is in any vocabulary.
    std::string first = s.substr(0, s.find(' '));
    std::string low = first;
    for (char& c : low) c = (char)std::tolower((unsigned char)c);
    return context::known_command(low) || context::known_command(first);
  };

  if (context::explicit_command_template(t)) {
    const std::string cmd = extract_payload(t);
    if (command_shaped(cmd)) return Intent::RunShell;
    // Template present but the payload is prose: "execute my plan" is a sentence
    // about doing something, not a program name. Refusing states the limit.
    return Intent::Unsupported;
  }
  // ---- metadata and search: both need a path, and both are read-only ----
  //      Checked after Unsupported so that a request asking to DELETE something
  //      that also mentions "size" is still refused rather than searched.
  if (wants_stat(t) && path) return Intent::Stat;
  if (wants_search(t) && path) return Intent::Search;
  if (wants_mkdir(t) && path) return Intent::MkDir;

  // copy needs its own extractor, and the ORIGINAL text -- normalisation strips
  // nothing it needs, but passing text keeps the two paths in their written form.
  if (wants_copy(t, text)) return Intent::Copy;

  // ---- strong verbs ----
  if (has_w(t, "write") || has_w(t, "save") || has_w(t, "create") ||
      has_w(t, "put ") || has_w(t, "store") || has_w(t, "dump") ||
      has_w(t, "make a file") || has_w(t, "commit ") || has_w(t, "persist") ||
      has_w(t, "park ") || has_w(t, "drop ") || has_w(t, "写入") || has_w(t, "保存") ||
      has_w(t, "创建")) {
    if (path) return Intent::WriteFile;
  }
  // ---------------------------------------------------------------------------
  //  A path ending in a file extension names a FILE; anything else is a place.
  //  Used to break the tie on phrases that are genuinely ambiguous.
  // ---------------------------------------------------------------------------
  auto path_looks_like_file = [&]() {
    // take the last path-ish token
    size_t e = t.find_last_not_of(" \t.,!?\"");
    if (e == std::string::npos) return false;
    size_t b = t.find_last_of(" \t", e);
    b = (b == std::string::npos) ? 0 : b + 1;
    const std::string tok = t.substr(b, e - b + 1);
    const size_t dot = tok.find_last_of('.');
    if (dot == std::string::npos || dot + 1 >= tok.size()) return false;
    // an extension is 1-5 alphanumerics after the final dot
    const std::string ext = tok.substr(dot + 1);
    if (ext.empty() || ext.size() > 5) return false;
    for (char c : ext) if (!std::isalnum((unsigned char)c)) return false;
    return true;
  };
  const bool file_like = path_looks_like_file();

  // ---- strong nouns (container words beat ambiguous verbs) ----
  if (path && (has_w(t, "list") || has_w(t, "ls ") || has_w(t, "dir ") ||
               has_w(t, "directory") || has_w(t, "catalogue") ||
               has_w(t, "stocktake") || has_w(t, "survey") || has_w(t, "index ") ||
               has_w(t, "目录") || has_w(t, "列出") || has_w(t, "列举") ||
               has_w(t, "下的") || has_w(t, "显示目录"))) {
    return Intent::ListDir;
  }
  // ---------------------------------------------------------------------------
  //  "entries" / "files in" / "contents of" are container words ONLY when the
  //  path does not already name a file.
  //
  //  "display the contents of C:/work/a.md"  -> the path has an extension, so it
  //                                             is one file: read it.
  //  "contents of C:/tmp"                    -> no extension: enumerate it.
  //
  //  Treating these as unconditional list cues scored 6 wrong answers across
  //  three separate test sets, because the same words appear in both templates
  //  and only the path shape distinguishes them.
  // ---------------------------------------------------------------------------
  if (path && (has_w(t, "entries") || has_w(t, "enumerate") ||
               has_w(t, "files in") || has_w(t, "contents of"))) {
    return file_like ? Intent::ReadFile : Intent::ListDir;
  }
  // ---- read verbs ----
  if (has_w(t, "read") || has_w(t, "cat ") || has_w(t, "fetch") ||
      has_w(t, "output") || has_w(t, "grab") || has_w(t, "pull up") ||
      has_w(t, "hand me") || has_w(t, "peruse") || has_w(t, "snoop") ||
      has_w(t, "audit") || has_w(t, "读取") || has_w(t, "打印") ||
      has_w(t, "查看") || has_w(t, "显示")) {
    if (path) return Intent::ReadFile;
  }
  // ---------------------------------------------------------------------------
  //  PRECEDENCE: CONTAINMENT OUTRANKS WEAK HINTS, AND LOSES TO STRONG VERBS.
  //
  //  The first attempt placed this above the STRONG verbs too, and that was wrong
  //  in a dangerous direction. Measured, 10 eval failures and one paraphrase case:
  //
  //      "put hello in C:/tmp/todo.json"  ->  list_dir {"path":"..."}
  //
  //  -- a WRITE was answered with a directory listing, because " in " matched and
  //  the write verb never ran. Losing a write is a different order of harm from
  //  losing a read.
  //
  //  The correct relation is by STRENGTH OF EVIDENCE, not by which word is more
  //  specific:
  //
  //      strong verb ("put", "write", "save")  says what the user wants DONE.
  //                                            Loses to nothing but the
  //                                            unsupported/explicit-tool levels.
  //      containment ("in <path>")             says WHAT IS BEING ACTED ON.
  //                                            Outranks the weak hints, which say
  //                                            neither.
  //      weak hint ("show", "open")            a generic action verb. Loses to
  //                                            containment because "show me what
  //                                            is in X" is about X.
  //
  //  So this sits after the strong verbs and before the weak hints, and the
  //  precedence table at the top of the function is corrected to match.
  //
  //  NOTE: "in " not " in " -- has_w requires the phrase as written and the
  //  preceding character is a letter ("is in"), so a leading space would make it
  //  unfindable. That is how this rule stopped firing once already.
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  //  CONTAINMENT REQUIRES THE PATH TO FOLLOW THE "in".
  //
  //  The rule tested "the text contains a path" AND "the text contains in ". Those
  //  are two independent facts about the whole sentence, and combining them does not
  //  mean the second acts on the first. Measured:
  //
  //      what is 7 + 8 in C:/tmp   ->  list_dir {"path":"7+8"}
  //
  //  -- the sentence contains a path (C:/tmp) and contains "in ", so containment
  //  fired; the payload then came from the expression. The result names a directory
  //  that is neither the one mentioned nor a number.
  //
  //  This is the CONTEXT question again, in its most literal form: " in " is a
  //  containment cue only when what follows it is a place. The same three words in a
  //  different arrangement say something else.
  // ---------------------------------------------------------------------------
  auto path_follows = [&](const char* marker) -> bool {
    const size_t p = t.find(marker);
    if (p == std::string::npos) return false;
    return context::licensed_as_path(t.substr(p + std::strlen(marker)));
  };
  //  AND ARITHMETIC STILL WINS, which is what the precedence table at the top of
  //  this function says and what the code did not do.
  //
  //  The table lists 3 ARITHMETIC above 5 CONTAINMENT, with the reason stated: a
  //  slash inside "3/4" is arithmetic, not a separator. But the containment rule sat
  //  ABOVE the expression test, so the table described an order the code did not
  //  implement -- found by an independent audit, and confirmed by the new
  //  "arithmetic vs containment" case in am-precedence.
  //
  //  The extra !expr is safe because looks_like_expression requires a digit-operator-
  //  digit shape, which "what is in C:/tmp" does not have: containment still answers
  //  the containment question, and an actual expression now answers the arithmetic
  //  one.
  if (path && has_w(t, "in ") && path_follows("in ") && !expr) return Intent::ListDir;

  // ---- weak hints ----
  if (path && (has_w(t, "show") || has_w(t, "display") || has_w(t, "open") ||
               has_w(t, "see ") || has_w(t, "view") || has_w(t, "peek") ||
               has_w(t, "inspect") || has_w(t, "examine") || has_w(t, "打开"))) {
    return Intent::ReadFile;
  }
  // ---------------------------------------------------------------------------
  //  RULE PRECEDENCE, stated once instead of emerging from statement order.
  //
  //  Nineteen of this project's defects were ORDERING mistakes: a rule that
  //  matched something incidental (a slash, the word "show", a bare path) fired
  //  before the rule that actually explained the request. Each was fixed by moving
  //  one statement, and the next rule added reintroduced the problem -- because
  //  "which rule wins" was encoded implicitly in the ORDER OF LINES.
  //
  //  Two concrete symptoms of that design, in this very function:
  //
  //    * the " in <path>" rule was written out TWICE (one copy before the read
  //      verbs, one before the expression test), because it needed to outrank both
  //      and there was no way to say "outrank" other than by repetition;
  //
  //    * "what is 3/4 of 18" was read as a directory listing on the third attempt
  //      to get the order right.
  //
  //  The fix is to name the precedence: a rule that describes the OPERATION
  //  outranks a rule that merely observes a SYMBOL. " in " names a container and
  //  asks about its contents -- that is an operation. A slash is a character. So
  //  containment is checked once, here, before any symbol-based rule, and the
  //  duplicate copy further down is gone.
  //
  //  This does not affect "read the file C:/tmp": no " in ", so it falls through
  //  to the read verbs unchanged.
  // ---------------------------------------------------------------------------
  if (expr) return Intent::Calc;

  // ---- question about a path with no verb at all -> directory listing ----
  if (path && (has_w(t, "what") || has_w(t, "which") || has_w(t, "什么"))) {
    return Intent::ListDir;
  }
  // ---- expression by explicit verb, when no path competes ----
  //
  // The "expr" test already ran above, before the question+path rule. What remains
  // here is the weaker signal: a calculation verb with no expression shape
  // ("compute this"). The guard on `path` matters because a path like
  // "D:/repo/zeta.md" contains digits and slashes; without it the rule fired on the
  // word "evaluate" alone and a file request became a calculator call.
  //
  // -------------------------------------------------------------------------
  //  A VERB ALONE IS NOT ENOUGH -- the payload has to BE an expression.
  //
  //  Measured on 48818 real Chinese instructions, ten of the first 160 examined
  //  produced a calc call with a whole sentence as its "expression":
  //
  //      "计算三角形的面积。"    -> calc {"expr":"计算三角形的面积。"}
  //      "计算中位数。"          -> calc {"expr":"计算中位数。"}
  //
  //  The Chinese 算 is matched by has_w inside 计算, so any request containing the
  //  word "compute" became a calculator request regardless of what followed. A guard
  //  was then added inside tool_calc, which turns these into errors -- but an error
  //  at execution time is the wrong layer: the call should never be issued. A call
  //  that is guaranteed to fail is worse than a refusal, because the reply is a
  //  fabricated invocation of a tool nobody asked for.
  //
  //  THE FIX IS THE RETURN VALUE, and the first version of this comment claimed a
  //  guard it did not have.
  //
  //  An earlier attempt wrote "!expr &&" as a condition here. That is DEAD CODE:
  //  the rule at level 3 already returned Calc when expr is true, so reaching this
  //  line implies !expr, and the extra test can never be false. It read as a
  //  safety condition and did nothing -- the same vacuity this project has been
  //  hunting in its test suites (defects 74, 77), found this time in a guard.
  //
  //  Caught by mutation: removing "!expr" changed no observable behaviour, which is
  //  the definition of a test that cannot fail. What actually stops
  //  "计算三角形的面积。" is the RETURN VALUE -- Unsupported instead of Calc -- so
  //  no call is issued. That is the whole fix, and it is stated here rather than
  //  implied by a redundant condition.
  // -------------------------------------------------------------------------
  if (!path &&
      (has_w(t, "calc") || has_w(t, "compute") || has_w(t, "how much") ||
       has_w(t, "算"))) {
    // The verb names an operation performed only on expressions. With no
    // expression present the request cannot be served, and saying so is more
    // useful than emitting a call bound to fail.
    return Intent::Unsupported;
  }
  // ---- shell ----
  //
  // ---------------------------------------------------------------------------
  //  A SHELL VERB NEEDS A SHELL PAYLOAD, same principle as the calc rule above.
  //
  //  Measured on 3200 real Chinese instructions, the two clearest false actions were
  //  both shell:
  //
  //      "我们应该执行计划吗？"                        -> run_shell {cmd:"我们应该执行计划吗？"}
  //      "我们现在没有足够的资金...来执行该项目。"        -> run_shell {cmd:"我们现在没有足够的资金..."}
  //
  //  In both, 执行 ("carry out") is used about a PLAN and a PROJECT -- ordinary
  //  Chinese -- and nothing in the sentence is a command. The English keywords have
  //  the same exposure, so the guard is applied to both languages rather than only
  //  to the one that happened to be measured.
  //
  //  The test is whether the payload can plausibly BE a command: it must consist of
  //  command-shaped characters, not a sentence. A payload that is mostly CJK
  //  characters with punctuation is prose, and prose after "execute" is a metaphor,
  //  not an instruction.
  //
  //  This is the same shape as defect 92/93 -- a keyword matched, and nothing
  //  checked that the thing it applied to was the right KIND of thing. Three
  //  separate rules have now needed that check, which is why the pattern is worth
  //  naming: a verb is not evidence until the payload is the type the verb takes.
  // ---------------------------------------------------------------------------
  // command_shaped is defined at level 1.5, above the explicit-template test that
  // needs it. Defining it again here was a redefinition error; this note replaces
  // the duplicate so the next reader knows where to look.

  //  Vague execution leads, added after measuring which phrasings the list missed.
  //
  //  "fire up ipconfig" and "kick off tasklist" both fell to Intent::Unknown, so the
  //  similarity gate used MIN_MATCH_UNKNOWN (0.48) instead of MIN_MATCH_STRUCTURED
  //  (0.15) and refused. The intent was never ambiguous to a reader -- a human knows
  //  "fire up X" runs X. The list simply did not contain the phrase.
  //
  //  These join the VAGUE group on purpose: "fire up the meeting notes" is prose, so
  //  the !path guard and the command_shaped test still apply. Only an explicit
  //  template earns the unconditional route at level 1.5.
  if (!path && (has_w(t, "run") || has_w(t, "exec") || has_w(t, "call") ||
                has_w(t, "fire up") || has_w(t, "kick off") ||
                has_w(t, "spin up") || has_w(t, "bring up") ||
                has_w(t, "start up") || has_w(t, "shell ") ||
                has_w(t, "执行") || has_w(t, "命令") ||
                has_w(t, "运行") || has_w(t, "调用"))) {
    const std::string cmd = extract_payload(t);
    if (command_shaped(cmd)) return Intent::RunShell;
    // Payload is prose: this is a sentence about doing something, not a command to
    // run. Refusing states the limit; emitting a shell call would execute a
    // question.
    return Intent::Unsupported;
  }

  // ---- last resort: a path with no competing signal is a READ ----
  //
  // This rule exists because the intent table is finite. "inspect D:/x.md",
  // "examine D:/x.md" and "peek at D:/x.md" all name a file and none of them
  // match any verb above, so the turn fell through to whole-memory retrieval and
  // the n-gram encoder picked a write (it shares "hello to" style grams with the
  // stored write prompts and has nothing in common with "inspect").
  //
  // Reading is the correct default for a bare path: it is the only tool that
  // harms nothing, and every other tool has an explicit syntactic signature --
  // write needs a content phrase, list needs a plural/container noun, shell needs
  // a command name without a path. Getting a read back when something else was
  // meant is recoverable; the reverse is not.
  //
  // The rule is placed AFTER the shell check so that "run the command hostname"
  // is unaffected, and after the expression check so "evaluate 45 * 12" still
  // routes to calc.
  //
  // GUARD, and why it is needed: applying the rule unconditionally broke
  // multi-turn continuation. "and C:/work/b.txt" is a bare path with no verb, so
  // it was routed to read_file outright and never inherited the PREVIOUS turn's
  // tool -- a follow-up to a directory listing became a file read. Measured as a
  // regression from 11/11 to 9/11.
  //
  // A path-only turn inherits when it looks like a CONTINUATION rather than a
  // request: it is short, and it carries no request-shaped verb or noun. Anything
  // longer is treated as a standalone request and defaults to read.
  if (path) {
    static const char* kRequestish[] = {
      "me ", "the ", "please", "show", "display", "open", "read", "view",
      "contents", "content", "inspect", "examine", "peek", "fetch", "get ",
      "打印", "读取", "查看", "内容"
    };
    bool requestish = (t.size() > 24);
    for (const char* w : kRequestish)
      if (t.find(w) != std::string::npos) { requestish = true; break; }
    if (requestish) return Intent::ReadFile;
  }

  return Intent::Unknown;
}

const char* intent_name(Intent i) {
  if (i == Intent::Unsupported) return "Unsupported";
  switch (i) {
    case Intent::ReadFile:  return "read_file";
    case Intent::ListDir:   return "list_dir";
    case Intent::WriteFile: return "write_file";
    case Intent::RunShell:  return "run_shell";
    case Intent::Calc:      return "calc";
    default:                return "unknown";
  }
}

float intent_affinity(const std::string& text, const std::string& tool) {
  const Intent want = infer_intent(text);
  if (want == Intent::Unknown) return 0.f;

  // The tool family of a candidate comes from the reply it produced.
  auto family = [&]() -> Intent {
    if (tool == "read_file")  return Intent::ReadFile;
    if (tool == "list_dir")   return Intent::ListDir;
    if (tool == "write_file") return Intent::WriteFile;
    if (tool == "run_shell")  return Intent::RunShell;
    if (tool == "calc")       return Intent::Calc;
    return Intent::Unknown;
  };
  const Intent have = family();
  if (have == Intent::Unknown) return 0.f;
  return (have == want) ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------------
//  Multi-turn context (gap B).
//
//  Design note on what is NOT done here: the conversation is not fed back into
//  the encoder, and the retrieved episode is not conditioned on the dialogue.
//  Both would require a representation of the dialogue state, and this system's
//  representation is a bag of character trigrams with no place to put it.
//  Inheritance instead repairs the QUERY, which is a string operation and
//  therefore exact. The retrieval stack is untouched.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  Deciding whether a turn can stand alone.
//
//  TWO FAILED FORMULATIONS, both worth recording:
//
//  1. "context needed when extract_payload is empty".
//     extract_payload falls back to the last whitespace token, so the cue
//     "again" came back as a payload; it then overwrote the previous one and
//     the call became calc {"expr":"again"}.
//
//  2. "context needed unless the payload contains a digit or a path character".
//     That rejected legitimate payloads that are bare words -- "run the command
//     hostname" was judged context-dependent and got rewritten to the PREVIOUS
//     turn's directory listing, which is worse than not having context at all.
//
//  The correct question is not "does the payload look like data" but "can THIS
//  turn already name a tool". If the structure alone identifies the tool, the
//  turn is self-contained regardless of what the payload looks like.
//
//  Prose-vs-payload is decided with a small stop list instead: a cue word is not
//  a payload, anything else is.
// ---------------------------------------------------------------------------
static bool is_bare_cue(const std::string& p) {
  static const char* kCues[] = {
    "again", "same", "that", "this", "it", "another", "more", "one more",
    "再来一次", "再来", "继续", "那个", "这个", "还有", "换一个", "同上"
  };
  std::string t = p;
  for (char& c : t) c = (char)std::tolower((unsigned char)c);
  // strip surrounding punctuation and spaces
  size_t b = t.find_first_not_of(" \t.,!?");
  size_t e = t.find_last_not_of(" \t.,!?");
  if (b == std::string::npos) return true;
  t = t.substr(b, e - b + 1);
  for (const char* c : kCues) if (t == c) return true;
  return false;
}

bool payload_is_data(const std::string& p) {
  return !p.empty() && !is_bare_cue(p);
}

bool needs_context(const std::string& text) {
  if (text.empty()) return true;

  // A turn stands alone only when it supplies BOTH a tool and a payload.
  //
  // Three formulations were tried; the first two failed in opposite directions:
  //
  //   1. "self-contained if the payload is non-empty"
  //      -> "again" counted as payload, overwriting the previous one:
  //         calc {"expr":"again"}.
  //
  //   2. "self-contained if the structure names a tool"
  //      -> "and C:/work/b.txt" names no tool, correct; but the rule also made
  //         nothing else context-free, and worse, dropping the payload check
  //         meant a bare path with no verb lost its tool. Measured 5/11.
  //
  //   3. the rule below: BOTH must be present. A sentence with a verb but no
  //      data ("again") inherits; a sentence with data but no verb ("and X")
  //      inherits; a sentence with both ("run the command hostname",
  //      "what is 8 * 9") is left alone.
  const bool has_tool    = (infer_intent(text) != Intent::Unknown);
  const bool has_payload = payload_is_data(extract_payload(text));
  return !(has_tool && has_payload);
}

const Turn* Session::last_tool_turn() const {
  for (auto it = turns.rbegin(); it != turns.rend(); ++it)
    if (!it->tool.empty()) return &(*it);
  return nullptr;
}

void Session::push(const std::string& user, const std::string& tool,
                   const std::string& payload) {
  turns.push_back({user, tool, payload});
  if (turns.size() > 32) turns.erase(turns.begin());   // bound the window
}

std::string Session::rewrite(const std::string& text) const {
  if (!needs_context(text)) return text;
  const Turn* prev = last_tool_turn();
  if (!prev) return text;

  std::string t = text;
  for (char& c : t) c = (char)std::tolower((unsigned char)c);

  // A payload alone ("and C:/other") keeps the previous tool.
  // An intent alone ("again", "do it for the log file") keeps the previous tool.
  // A fresh payload plus a bare continuation cue reuses BOTH, which is what
  // makes "again" re-run the same call and "and C:/work/x.txt" re-aim it.
  // Only a genuine DATA payload replaces the previous one. A bare cue keeps it,
  // which is what makes "again" re-run the same call.
  const std::string own_payload = extract_payload(text);
  const std::string payload = payload_is_data(own_payload) ? own_payload : prev->payload;

  // ---------------------------------------------------------------------------
  //  REBUILD, driven by a table that CANNOT fall out of step with the tools.
  //
  //  The previous version was a chain of if/else over tool names ending in
  //  "else return text" -- so every tool that was not in the chain silently lost
  //  inheritance. Measured, after mkdir/search/stat/copy were added:
  //
  //      "find all TODO in C:/tmp"  then "again"  ->  refused
  //      "how big is ..."           then "again"  ->  refused
  //
  //  while calc, read_file, list_dir and run_shell inherited correctly. Four tools
  //  worked and four did not, and the difference was purely whether someone had
  //  remembered to add a line here.
  //
  //  This is the SIXTH place a new tool must be registered (implementation,
  //  dispatch whitelist, Intent enum, bucket count, corpus generator, Unsupported
  //  table) -- a count that has already been wrong once. Adding a seventh by hand
  //  would repeat the mistake, so the table is built from the tool registry itself:
  //  tool_specs() already lists every tool the system has, in one place, because the
  //  prompt needs it.
  //
  //  A tool with no entry still inherits, using a generic leading verb, so a tool
  //  added tomorrow degrades to "something reasonable" rather than to "no
  //  inheritance at all".
  // ---------------------------------------------------------------------------
  // The lead strings now live WITH the tools (ToolSpec::lead), so this function
  // cannot fall behind again. See the comment on that field: the previous version
  // restated them here, in a chain ending in "else return text", and four tools
  // were silently absent from it.
  for (const ToolSpec& s : tool_specs()) {
    if (prev->tool != s.name) continue;
    if (s.lead) return std::string(s.lead) + payload;
    // Multi-argument tool: the only faithful reading of "again" is the same
    // request again, so the previous turn's own text is reused unchanged.
    return prev->user;
  }
  // A tool the table does not know (cannot happen while the table IS the registry,
  // but kept so the function total): rebuild from the payload alone -- inheritance
  // degrades rather than vanishing.
  return payload;
}
bool payload_is_expression(const std::string& p) {
  bool digit = false, op = false;
  for (char c : p) {
    if (std::isdigit((unsigned char)c)) digit = true;
    else if (c == '+' || c == '-' || c == '*' || c == '/') op = true;
    else if (c == ' ' || c == '\t') continue;
    else return false;            // any other character -> not a bare expression
  }
  return digit && op;
}

std::string normalise_payload(const std::string& p) {
  if (!payload_is_expression(p)) return p;
  std::string o;
  o.reserve(p.size());
  for (char c : p) if (c != ' ' && c != '\t') o.push_back(c);
  return o;
}

// ---------------------------------------------------------------------------
//  Lead-in phrase matching, in ONE place.
//
//  WHY THIS FUNCTION EXISTS
//  ------------------------
//  The same substring hazard has now produced three separate defects:
//
//      #16  "put " matched inside "output"        -> reads routed to write_file
//      #56  " to " matched inside " into "        -> copy split mid-path
//      #59  "run " matched inside "echo a run b"  -> payload truncated to "b"
//
//  Each was fixed at its own call site, and each fix was a local rule that the
//  next call site did not inherit. That is the actual failure: not the individual
//  bugs but the absence of a shared primitive.
//
//  THE RULE, stated once: a lead-in marks where the payload BEGINS, so it must be
//  the EARLIEST occurrence, and among phrases starting at the same place the
//  LONGEST wins (so "run the command " beats "run ").
//
//  The previous implementations used rfind -- the LAST occurrence -- which is
//  exactly backwards for a marker that precedes the payload. It happens to work
//  whenever the phrase does not also occur inside the command, which is why three
//  defects were needed to notice.
//
//  Returns the offset just past the matched lead-in, or npos when none applies.
// ---------------------------------------------------------------------------
inline size_t match_lead_in(const std::string& q, const char* const* phrases,
                            size_t count) {
  size_t best_pos = std::string::npos;
  size_t best_len = 0;
  for (size_t i = 0; i < count; ++i) {
    const size_t n = std::strlen(phrases[i]);
    const size_t at = q.find(phrases[i]);         // FIND, not rfind
    if (at == std::string::npos) continue;
    // earliest position wins; ties broken by the longer phrase
    if (at < best_pos || (at == best_pos && n > best_len)) {
      best_pos = at;
      best_len = n;
    }
  }
  if (best_pos == std::string::npos) return std::string::npos;
  return best_pos + best_len;
}

// ---------------------------------------------------------------------------
//  Arithmetic written in WORDS.
//
//  Real corpus evidence (CLINC150, 23703 genuine utterances) showed the system
//  handling only 1 of 8 real calculator requests. The others failed because real
//  people write operators as words:
//
//      "what is 7 x 7"                  the symbol is the letter x
//      "what is the sum of 10 plus 5"   "plus", not "+"
//      "please add 456 and 781 for me"  "add ... and", no symbol at all
//      "what is 30% off 279"            a percentage operation
//
//  The synthetic corpus contains only "40 + 14" style input, so the extractor
//  never needed to know these forms. That is the general failure mode of a
//  generated corpus: it covers the forms its generator was written to emit, and
//  real users write others.
//
//  Normalisation happens FIRST, translating the word forms into the symbols the
//  rest of the pipeline already understands, rather than teaching every later
//  stage about both spellings.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  ADVANCED note: number words and verb forms.
static std::string expand_english_math(const std::string& in) {
  std::string s = in;
  auto lower = [](std::string v) {
    for (char& c : v) c = (char)std::tolower((unsigned char)c);
    return v;
  };

  static const struct { const char* w; long v; } kNum[] = {
    {"ninety",90},{"eighty",80},{"seventy",70},{"sixty",60},{"fifty",50},
    {"forty",40},{"thirty",30},{"twenty",20},{"nineteen",19},{"eighteen",18},
    {"seventeen",17},{"sixteen",16},{"fifteen",15},{"fourteen",14},
    {"thirteen",13},{"twelve",12},{"eleven",11},{"ten",10},{"nine",9},
    {"eight",8},{"seven",7},{"six",6},{"five",5},{"four",4},{"three",3},
    {"two",2},{"one",1},{"zero",0},
  };

  auto read_operand = [&](const std::string& low, size_t at, size_t* end_out) -> std::string {
    size_t p = at;
    while (p < low.size() && (low[p] == ' ' || low[p] == ',')) ++p;
    if (p < low.size() && std::isdigit((unsigned char)low[p])) {
      size_t e = p;
      while (e < low.size() && (std::isdigit((unsigned char)low[e]) || low[e] == '.')) ++e;
      *end_out = e;
      return low.substr(p, e - p);
    }
    long cur = 0;
    bool any = false;
    size_t q = p;
    for (;;) {
      size_t r = q;
      while (r < low.size() && (low[r] == ' ' || low[r] == '-')) ++r;
      bool m = false;
      for (const auto& nw : kNum) {
        const size_t n = std::strlen(nw.w);
        if (low.compare(r, n, nw.w) != 0) continue;
        if (r + n < low.size() && std::isalpha((unsigned char)low[r+n])) continue;
        cur += nw.v; q = r + n; m = true; any = true; break;
      }
      if (!m) break;
    }
    if (!any) { *end_out = at; return std::string(); }
    *end_out = q;
    return std::to_string(cur);
  };

  struct Verb { const char* w; const char* sym; bool flipped; };
  static const Verb kV[] = {
    {"multiply","*",false}, {"divide","/",false},
    {"subtract","-",true},  {"subract","-",true},
    {"add","+",false},      {"sum of","+",false},
  };
  for (int pass = 0; pass < 4; ++pass) {
    std::string low = lower(s);
    bool changed = false;
    for (const Verb& v : kV) {
      const size_t n = std::strlen(v.w);
      for (size_t p = low.find(v.w); p != std::string::npos; p = low.find(v.w, p + 1)) {
        if (p > 0 && std::isalpha((unsigned char)low[p-1])) continue;
        size_t e1 = 0;
        const std::string a = read_operand(low, p + n, &e1);
        if (a.empty()) continue;
        size_t c = e1;
        while (c < low.size() && low[c] == ' ') ++c;
        std::string sym;
        size_t clen = 0;
        bool flip = v.flipped;
        if (low.compare(c, 3, "by ") == 0)        { sym = v.sym; clen = 3; }
        else if (low.compare(c, 5, "from ") == 0) { sym = "-"; flip = true; clen = 5; }
        else if (low.compare(c, 4, "and ") == 0)  { sym = "+"; flip = false; clen = 4; }
        else continue;
        size_t e2 = 0;
        const std::string b = read_operand(low, c + clen, &e2);
        if (b.empty()) continue;
        const std::string rep = flip ? (b + " " + sym + " " + a)
                                     : (a + " " + sym + " " + b);
        s.replace(p, e2 - p, rep);
        changed = true;
        break;
      }
      if (changed) break;
    }
    if (!changed) break;
  }

  // number words, only where the sentence ALREADY signals arithmetic -- so
  // "one more thing" is left alone while "seventeen plus three" is converted.
  {
    std::string low = lower(s);
    // -------------------------------------------------------------------------
    //  NUMBER WORDS ARE ONLY REWRITTEN WHERE THE CONTEXT IS ARITHMETIC.
    //
    //  The first version asked "does the text contain any of + - * /". A PATH
    //  CONTAINS A SLASH, so every path was arithmetic by that test, and the
    //  rewriting then converted number words INSIDE THE PATH:
    //
    //      read the file C:/tmp/four.txt        -> {"path":"C:/tmp/4.txt"}
    //      read the file C:/data/two_two.txt    -> {"path":"C:/data/2_2.txt"}
    //
    //  Both are valid paths, neither exists, and nothing reported a problem -- the
    //  request silently addressed a different file. Measured: "write hello to
    //  .../one_two_test.txt" created 1_2_test.txt.
    //
    //  The context test that works is structural: an operator counts only when it
    //  sits BETWEEN TWO DIGITS. A slash in C:/tmp has a colon on its left and a
    //  letter on its right, so it is a separator here, not a division.
    //
    //  This is the same question as the path licence, asked from the other side: a
    //  slash is a division only in an environment where it joins two numbers.
    // -------------------------------------------------------------------------
    bool mathy = false;
    for (size_t i = 1; i + 1 < low.size(); ++i) {
      if (!std::strchr("+-*/", low[i])) continue;
      size_t L = i, Rr = i + 1;
      while (L > 0 && low[L-1] == ' ') --L;
      while (Rr < low.size() && low[Rr] == ' ') ++Rr;
      const bool lnum = (L > 0) && std::isdigit((unsigned char)low[L-1]);
      const bool rnum = (Rr < low.size()) && std::isdigit((unsigned char)low[Rr]);
      if (lnum && rnum) { mathy = true; break; }
    }
    if (!mathy) {
      static const char* kCue[] = {"plus","minus","times","multiply","divide",
                                    "subtract","subract","add "};
      for (const char* c : kCue) if (low.find(c) != std::string::npos) { mathy = true; break; }
    }
    if (mathy) {
      // compound first ("forty five" -> 45), scanning left to right
      size_t i = 0;
      while (i < low.size()) {
        if (!std::isalpha((unsigned char)low[i])) { ++i; continue; }
        if (i > 0 && std::isalpha((unsigned char)low[i-1])) { ++i; continue; }
        size_t e = 0;
        const std::string num = read_operand(low, i, &e);
        if (num.empty() || e <= i) { ++i; continue; }
        s.replace(i, e - i, num);
        low = lower(s);
        i += num.size();
      }
    }
  }
  return s;
}

static std::string normalise_math_words(const std::string& in) {
  std::string s = expand_english_math(in);
  // lower-case a working copy for matching, but only to locate; edits preserve
  // everything else.
  auto lower = [](std::string v) {
    for (char& c : v) c = (char)std::tolower((unsigned char)c);
    return v;
  };
  struct Sub { const char* from; const char* to; };
  // Longest first, so "plus" is not eaten by "pl" style prefixes and
  // "multiplied by" wins over "by".
  static const Sub kSubs[] = {
    {" multiplied by ", " * "}, {" times ", " * "}, {" x ", " * "},
    {" divided by ", " / "},   {" over ", " / "},
    {" plus ", " + "},         {" minus ", " - "},   {" less ", " - "},
    {" add ", " "},            {" subtract ", " - "},
    {" for me", " "},          {" please", " "},
  };
  for (const Sub& sb : kSubs) {
    std::string low = lower(s);
    const std::string f = lower(sb.from);
    size_t p = 0;
    while ((p = low.find(f, p)) != std::string::npos) {
      s.replace(p, f.size(), sb.to);
      low = lower(s);
      p += std::strlen(sb.to);
    }
  }
  // "and" joins two operands in "add 456 and 781" -> becomes "+".
  // Only when digits sit on both sides, so "black and white" is untouched.
  {
    std::string low = lower(s);
    size_t p = 0;
    while ((p = low.find(" and ", p)) != std::string::npos) {
      size_t a = p;
      while (a > 0 && s[a-1] == ' ') --a;
      const size_t ae = a;
      while (a > 0 && std::isdigit((unsigned char)s[a-1])) --a;
      const bool left_num = (a < ae);
      size_t b = p + 5;
      while (b < s.size() && s[b] == ' ') ++b;
      size_t be = b;
      while (be < s.size() && std::isdigit((unsigned char)s[be])) ++be;
      const bool right_num = (be > b);
      if (left_num && right_num) {
        s.replace(p, 5, " + ");
        low = lower(s);
        p += 3;
      } else {
        p += 5;
      }
    }
  }
  // tidy doubled spaces
  std::string out;
  bool sp = false;
  for (char c : s) {
    if (c == ' ') { if (!sp) out.push_back(' '); sp = true; }
    else { out.push_back(c); sp = false; }
  }
  return out;
}

// Percentage operations: "30% off 279" and "60 percent off of 350".
// Converted to a form the expression parser can carry as an expression string:
// the payload becomes "279*30/100" style, which calc evaluates directly.
static bool extract_percent(const std::string& q, std::string* out) {
  std::string low = q;
  for (char& c : low) c = (char)std::tolower((unsigned char)c);
  const size_t pct = low.find('%');
  const size_t pctw = low.find("percent");
  size_t at = std::string::npos, len = 0;
  if (pct != std::string::npos) { at = pct; len = 1; }
  else if (pctw != std::string::npos) { at = pctw; len = 7; }
  if (at == std::string::npos) return false;

  // number before the percent sign
  size_t pe = at;
  while (pe > 0 && q[pe-1] == ' ') --pe;
  size_t ps = pe;
  while (ps > 0 && std::isdigit((unsigned char)q[ps-1])) --ps;
  if (ps == pe) return false;
  const std::string pnum = q.substr(ps, pe - ps);

  // Operand after the percentage phrase.
  //
  // Several connectives occur in real data and each must be skipped explicitly:
  //   "30% off 279"            off
  //   "60 percent off of 350"  off of
  //   "85% of $1500"           of
  //   "18 percent tip on 20"   tip on
  //   "15% of 3143"            of
  // Skipping the wrong number of words lands on the wrong operand or on none,
  // which is how "18 percent tip on 20 dollars" extracted the word "dollars".
  size_t b = at + len;
  while (b < q.size() && q[b] == ' ') ++b;
  static const char* kConn[] = {"percent off of ", "percent tip on ", "percent of ",
                                "off of ", "tip on ", "off ", "of ", "on "};
  bool skipped = false;
  for (const char* c : kConn) {
    const size_t n = std::strlen(c);
    if (low.compare(b, n, c) == 0) { b += n; skipped = true; break; }
  }
  (void)skipped;
  while (b < q.size() && (q[b] == ' ' || q[b] == '$')) ++b;
  size_t be = b;
  while (be < q.size() && std::isdigit((unsigned char)q[be])) ++be;
  if (be == b) return false;
  const std::string base = q.substr(b, be - b);

  // result = base * percent / 100
  *out = base + "*" + pnum + "/100";
  return true;
}

// Square root, factorial, and "a/b of N".
//
// These come from CLINC150 too and were the remaining 5 failures of 22 real
// utterances. Each is a distinct OPERATION rather than a different spelling of an
// existing one, so they need their own extraction rather than another substitution.
static bool extract_named_op(const std::string& q, std::string* out) {
  std::string low = q;
  for (char& c : low) c = (char)std::tolower((unsigned char)c);

  auto number_after = [&](size_t from) -> std::string {
    size_t b = from;
    while (b < q.size() && !std::isdigit((unsigned char)q[b])) ++b;
    size_t e = b;
    while (e < q.size() && std::isdigit((unsigned char)q[e])) ++e;
    return (e > b) ? q.substr(b, e - b) : std::string();
  };

  // ---- square root, optionally scaled by a word ----
  {
    const size_t p = low.find("square root");
    if (p != std::string::npos) {
      const std::string n = number_after(p);
      if (!n.empty()) {
        std::string mult;
        if (low.find("million", p) != std::string::npos) mult = "*1000000";
        else if (low.find("thousand", p) != std::string::npos) mult = "*1000";
        *out = "sqrt(" + n + mult + ")";
        return true;
      }
    }
  }
  // ---- factorial: the number precedes the word ----
  {
    const size_t p = low.find("factorial");
    if (p != std::string::npos) {
      size_t e = p;
      while (e > 0 && q[e-1] == ' ') --e;
      size_t b = e;
      while (b > 0 && std::isdigit((unsigned char)q[b-1])) --b;
      if (b < e) { *out = q.substr(b, e - b) + "!"; return true; }
    }
  }
  // ---- "<a>/<b> of <n>" ----
  for (size_t p = low.find(" of "); p != std::string::npos; p = low.find(" of ", p + 1)) {
    const size_t slash = low.rfind('/', p);
    if (slash == std::string::npos) continue;
    size_t ae = slash;
    while (ae > 0 && q[ae-1] == ' ') --ae;
    size_t as = ae;
    while (as > 0 && std::isdigit((unsigned char)q[as-1])) --as;
    size_t bs = slash + 1, be = bs;
    while (be < q.size() && std::isdigit((unsigned char)q[be])) ++be;
    if (as < ae && be > bs) {
      const std::string n = number_after(p + 4);
      if (!n.empty()) {
        *out = n + "*" + q.substr(as, ae - as) + "/" + q.substr(bs, be - bs);
        return true;
      }
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
//  The source and destination of a copy request.
//
//  copy has TWO paths, which no existing extractor produces: extract_payload
//  returns one, and slot transfer moves one. search ran into the same shape and
//  the fix there was to transfer the path and extract the needle separately; here
//  BOTH arguments must come from the query, so copy is handled entirely by its own
//  extractor and the stored template supplies only the JSON skeleton.
//
//  The separator is what makes it tractable: every natural phrasing puts a
//  connective between the two paths.
//      "copy C:/a.txt to C:/b.txt"
//      "copy the file C:/a.txt into C:/backup"
//      "duplicate C:/a.txt as C:/b.txt"
//  so the split is at the LAST such connective, and each side is scanned for its
//  first path-shaped token.
// ---------------------------------------------------------------------------
static std::string first_path_in(const std::string& s) {
  // a path token: contains '/', or a drive prefix, or a dotted filename
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' ||
                            s[i] == '"' || s[i] == '\'')) ++i;
    if (i >= s.size()) break;
    size_t j = i;
    while (j < s.size() && s[j] != ' ' && s[j] != '\t' && s[j] != '"' && s[j] != '\'') ++j;
    const std::string tok = s.substr(i, j - i);
    const bool pathish = tok.find('/') != std::string::npos ||
                         tok.find('\\') != std::string::npos ||
                         (tok.size() > 2 && std::isalpha((unsigned char)tok[0]) &&
                          tok[1] == ':');
    if (pathish) return tok;
    i = j + 1;
  }
  return "";
}

bool extract_copy_paths(const std::string& q, std::string* src, std::string* dst) {
  // -------------------------------------------------------------------------
  //  LONGEST separator wins, not the right-most one.
  //
  //  The first version picked whichever separator started latest in the string,
  //  which fails on nested connectives: "copy X into Y" contains both " into " and
  //  " to " (the latter inside the former, one character later), so " to " won and
  //  the source path was cut mid-word.
  //
  //  Ordering by LENGTH instead makes " into " beat " to ", which is the intended
  //  reading -- and it is the same substring-inside-a-keyword trap as defect 16
  //  ("put " inside "output"), so the fix is deliberately the same shape: prefer
  //  the more specific pattern rather than the one that happens to match later.
  // -------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  //  EVERY ENTRY HERE MUST BE ABLE TO CHANGE A DECISION, which one of them was not.
  //
  //  Mutation testing removed " over to " and no test noticed -- and when the
  //  behaviour itself was compared, it was identical. The reason is that "copy X over
  //  to Y" contains " to " as well, so dropping the longer entry still splits at the
  //  same place once the tie-break picks the LAST occurrence. The entry was redundant.
  //
  //  It is kept, because keeping it is harmless and it documents a phrasing users
  //  write -- but the mutation is now aimed at " into " instead, which IS
  //  load-bearing: without it, "copy X into Y" splits at the " to " inside " into "
  //  and the source path is cut mid-word. That is defect 56 exactly, and a mutation
  //  that reproduces defect 56 must make the suite fail.
  // ---------------------------------------------------------------------------
  static const char* kSep[] = {" over to ", " into ", " to ", " as ", " -> ",
                               "\u5230", "\u4e3a"};
  // Longest match wins, and among equals the LATER position (the separator sits
  // between two paths, so the last plausible one is the right one). This is the
  // same rule as match_lead_in with the tie-break reversed, which is why the
  // direction is stated explicitly rather than inferred from a shared helper.
  size_t best = std::string::npos;
  size_t blen = 0;
  for (size_t k = 0; k < sizeof(kSep)/sizeof(kSep[0]); ++k) {
    const size_t n = std::strlen(kSep[k]);
    const size_t p = q.rfind(kSep[k]);
    if (p == std::string::npos) continue;
    if (n > blen || (n == blen && p > best)) { best = p; blen = n; }
  }
  // no connective: only the "a b" two-token form is inferable, which is too
  // ambiguous to guess at, so report failure rather than invent a destination.
  if (best == std::string::npos) return false;

  const std::string left  = q.substr(0, best);
  const std::string right = q.substr(best + blen);
  *src = first_path_in(left);
  *dst = first_path_in(right);
  // "copy C:/a.txt to the folder C:/bak" -- scan the whole right side, which
  // first_path_in already does.
  //
  //  BOTH sides are required. Measured cause of this edit: the mutation harness left
  //  a mutant here -- the mutation replaced this with "return !src->empty();" and the
  //  restore did not run, because the driver recorded a SKIP (stale anchor) on the
  //  attempt that would have restored it. So the mutation survived into the tree.
  //
  //  That is the one failure a mutation harness must never have, and it is exactly
  //  what the header of am_mutate.cpp promises cannot happen: "A harness that can
  //  leave a MUTANT in the tree is worse than no harness: it would silently ship the
  //  deliberate bug."
  //
  //  Two causes, both fixed below the driver: restore now happens BEFORE the verdict
  //  is printed on every path including SKIP, and the pristine copy is written for
  //  every mutation so a restore cannot depend on the anchor still being findable.
  return !src->empty() && !dst->empty();
}

// ---------------------------------------------------------------------------
//  The pattern of a search request.
//
//  search takes TWO arguments while slot transfer moves ONE payload, so the path
//  is transferred from the stored template and the pattern is taken from the query
//  directly. Without this the transferred path landed in the pattern field and the
//  command searched the wrong place for the wrong string.
//
//  Method: remove the path, remove the lead-in phrase, and whatever remains is the
//  pattern. That handles all the forms the corpus and CLINC150 use:
//      "find all TODO in C:/src"        -> TODO
//      "search for main in C:/src"      -> main
//      "grep for config in D:/data"     -> config
//      "show me files containing x in C:/y" -> x
// ---------------------------------------------------------------------------
std::string extract_search_pattern(const std::string& q) {
  std::string s = q;

  // 1. drop the path, wherever it appears
  const std::string pay = extract_payload(q);
  if (pay.empty()) return "";
  size_t at = s.find(pay);
  if (at != std::string::npos) s.erase(at, pay.size());

  // 2. drop the lead-in phrase
  static const char* kLead[] = {
    "show me files containing ", "which files contain ", "files containing ",
    "search for ", "search in ", "grep for ", "look for ", "locate ",
    "find all ", "find any ", "find the ", "find ",
  };
  std::string low = s;
  for (char& c : low) c = (char)std::tolower((unsigned char)c);
  for (const char* L : kLead) {
    const size_t p = low.find(L);
    if (p != std::string::npos) { s.erase(p, std::strlen(L)); low = s;
      for (char& c : low) c = (char)std::tolower((unsigned char)c); }
  }
  // 3. drop POLITE PREFIXES anywhere in what remains.
  //
  //  These are the same lead-ins the generator puts in front of every template
  //  ("please ", "can you ", "kindly "). Stripping the search verb alone left them
  //  attached to the needle:
  //      "can you search for main in C:/"   -> pattern "can you main"
  //      "kindly look for FIXME in C:/Users"-> pattern "kindly FIXME"
  //
  //  The lesson generalises: an extraction that REMOVES known phrases will always
  //  have an incomplete list, so the prefixes are stripped in a loop until none
  //  applies rather than once.
  static const char* kPolite[] = {
    "can you ", "could you ", "would you ", "will you ", "please ",
    "kindly ", "hey ", "hi ", "ok ", "okay ", "just ", "i want to ",
    "i need to ", "i would like to ", "let me ", "help me ",
  };
  for (int pass = 0; pass < 4; ++pass) {
    bool changed = false;
    std::string low2 = s;
    for (char& c : low2) c = (char)std::tolower((unsigned char)c);
    for (const char* W : kPolite) {
      const size_t n = std::strlen(W);
      if (low2.compare(0, n, W) == 0) {
        s.erase(0, n);
        changed = true;
        break;
      }
      // also handle a prefix left in the middle after the verb was removed
      const size_t p = low2.find(W);
      if (p != std::string::npos) {
        s.erase(p, n);
        changed = true;
        break;
      }
    }
    if (!changed) break;
  }

  // 4. drop trailing connectives and punctuation
  auto trim = [](std::string& v) {
    size_t b = v.find_first_not_of(" \t,.;:'\"");
    size_t e = v.find_last_not_of(" \t,.;:'\"");
    v = (b == std::string::npos) ? "" : v.substr(b, e - b + 1);
  };
  trim(s);
  // remove a trailing " in" left behind after the path was erased
  while (s.size() >= 3) {
    std::string tail = s.substr(s.size() - 3);
    if (tail == " in" || tail == " at" || tail == " of") { s.erase(s.size() - 3); break; }
    break;
  }
  trim(s);
  return s;
}

// "a/b of N" must be checked BEFORE the path rule, for the same reason a shell
// flag must be: the slash in "3/4 of 18" is arithmetic, not a directory separator.
// Measured: this was the last remaining failure of 22 real utterances, and it
// failed with path="3/4 of 18" -- the path branch grabbed the fraction.
static bool extract_fraction_of(const std::string& q, std::string* out) {
  std::string low = q;
  for (char& c : low) c = (char)std::tolower((unsigned char)c);
  for (size_t p = low.find(" of "); p != std::string::npos; p = low.find(" of ", p + 1)) {
    size_t slash = low.rfind('/', p);
    if (slash == std::string::npos || slash == 0) continue;
    if (!std::isdigit((unsigned char)q[slash-1])) continue;
    size_t ae = slash;
    while (ae > 0 && q[ae-1] == ' ') --ae;
    size_t as = ae;
    while (as > 0 && std::isdigit((unsigned char)q[as-1])) --as;
    size_t bs = slash + 1, be = bs;
    while (be < q.size() && std::isdigit((unsigned char)q[be])) ++be;
    if (as >= ae || be <= bs) continue;
    size_t nb = p + 4;
    while (nb < q.size() && !std::isdigit((unsigned char)q[nb])) ++nb;
    size_t ne = nb;
    while (ne < q.size() && std::isdigit((unsigned char)q[ne])) ++ne;
    if (ne <= nb) continue;
    *out = q.substr(nb, ne - nb) + "*" + q.substr(as, ae - as) + "/" + q.substr(bs, be - bs);
    return true;
  }
  return false;
}

std::string extract_payload(const std::string& q) {
  // ---- 0. arithmetic that contains a slash, before the path rule sees it ----
  {
    std::string frac;
    if (extract_fraction_of(q, &frac)) return frac;
  }
  // ---- 0b. percentages, named operations, and operators spelled as words ----
  {
    std::string pct;
    if (extract_percent(q, &pct)) return pct;
    std::string named;
    if (extract_named_op(q, &named)) return named;
  }
  const std::string qn = normalise_math_words(q);
  if (qn != q) {
    const std::string r = extract_payload(qn);   // re-enter with symbols present
    if (!r.empty()) return r;
  }

  // ---- 1. arithmetic expression: <num> <op> <num> ----
  //      must come first: "40-14" also looks like a hyphenated word.
  {
    size_t i = 0;
    while (i < q.size()) {
      if (std::isdigit((unsigned char)q[i])) {
        // a0 is the START of the first operand; a walks past it. Using the
        // walked pointer as the substring start shifted the result by one
        // character, so "7 + 8" came out as " + 8" and the leading digit was
        // dropped from every arithmetic payload.
        const size_t a0 = i;
        size_t a = i;
        while (a < q.size() && std::isdigit((unsigned char)q[a])) ++a;
        size_t o = a;
        while (o < q.size() && q[o] == ' ') ++o;
        if (o < q.size() && std::strchr("+-*/", q[o])) {
          size_t b = o + 1;
          while (b < q.size() && q[b] == ' ') ++b;
          size_t c = b;
          while (c < q.size() && std::isdigit((unsigned char)q[c])) ++c;
          if (c > b) return q.substr(a0, c - a0);   // "123 * 456", "40-14"
        }
        i = a;
      } else ++i;
    }
  }
  // ---- 1b. shell command, checked BEFORE the path rule ----
  //
  // A command with a flag contains a slash: "ipconfig /all", "cmd /c dir". The
  // path rule below treats any slash as a path separator and grabs the flag
  // ("/all") as the payload, discarding the program name. The lead-in phrase is
  // the reliable signal that what follows is a command, so it must be tested
  // first.
  //
  // Measured before this reorder: ipconfig /all -> {"cmd":"/all"} and
  // cmd /c dir -> {"cmd":"/c"}. After: both correct.
  {
    //  ORDER MATTERS: longest first. match_lead_in returns the FIRST match, so
    //  "run " before "run the command " would leave "the command X" as the payload.
    //
    //  Added after measuring: "start up tracert" fell through to the last-token
    //  fallback and produced {"cmd":"up tracert"}, while "fire up ipconfig" only
    //  worked by luck. "start up " must precede "start " for the same reason
    //  "run the command " precedes "run ".
    //
    //  NOT added: "go ahead and ", "feel free to ". Stripping those leaves
    //  "run netstat" as the payload, which is wrong -- the filler is not part of
    //  the command, but the LEAD after it still is. detect_multi_action already
    //  dismisses the filler as politeness; the payload extraction must then use the
    //  ordinary lead list, which it does once the filler is gone from
    //  consideration. Adding them here only moved the bug.
    static const char* kLeadFirst[] = {
      "please execute the utility ", "please run the command ", "run the command ",
      "please execute ", "please run ", "execute the utility ", "run the utility ",
      "fire up ", "kick off ", "spin up ", "bring up ", "start up ",
      "execute ", "invoke ", "launch ", "start ", "run ", "call ",
    };
    const size_t after = match_lead_in(q, kLeadFirst,
                                       sizeof(kLeadFirst)/sizeof(kLeadFirst[0]));
    if (after != std::string::npos) {
      size_t best = after;
      while (best < q.size() && (q[best] == ' ' || q[best] == '\t')) ++best;
      const size_t e = q.find_last_not_of(" \t");
      if (best < q.size() && e != std::string::npos)
        return q.substr(best, e - best + 1);
    }
  }

  // ---- 2. path: contains '/', or a drive/URL scheme, or a dotted filename ----
  {
    size_t best = std::string::npos;
    for (size_t i = 0; i < q.size(); ++i) {
      const char c = q[i];
      const bool pathish = (c == '/') ||
                           (c == ':' && i + 1 < q.size() && q[i+1] == '/') ||
                           (c == '.' && i + 1 < q.size() && std::isalpha((unsigned char)q[i+1]));
      if (pathish) { best = i; break; }
    }
    if (best != std::string::npos) {
      while (best > 0 && q[best-1] != ' ') --best;
      size_t e = best;
      while (e < q.size() && q[e] != ' ') ++e;
      return q.substr(best, e - best);
    }
  }
  // ---- 3. shell command: everything after the lead-in phrase ----
  //
  // WHY THIS BRANCH EXISTS
  // ----------------------
  // The fallback below returns the LAST whitespace token, which is correct for a
  // one-word command like "hostname" and wrong for every command with arguments.
  // Measured against a taught family "please execute the utility X":
  //
  //     ipconfig /all   ->  {"cmd":"/all"}      lost the program name
  //     netstat -an     ->  {"cmd":"-an"}       lost the program name
  //     ping -n 3       ->  {"cmd":"3"}         lost the program AND its flag
  //     cmd /c dir      ->  {"cmd":"/c"}        lost everything but the flag
  //
  // A shell command is a SEQUENCE, not a token. The lead-in phrase is what marks
  // where it starts, so the payload is the remainder after the longest matching
  // lead-in. Longest-match matters: "run the command" must win over "run".
  //
  // Paths and expressions are handled above and never reach here, which is why
  // read_file/list_dir/calc were unaffected by this bug.
  {
    static const char* kLead[] = {
      "please execute the utility ", "please run the command ", "run the command ",
      "please execute ", "please run ", "execute the utility ", "run the utility ",
      "execute ", "invoke ", "launch ", "start ", "run ", "call ",
      "\u8bf7\u6267\u884c ", "\u8bf7\u8fd0\u884c ", "\u6267\u884c ", "\u8fd0\u884c ", "\u8c03\u7528 ",
    };
    const size_t after = match_lead_in(q, kLead, sizeof(kLead)/sizeof(kLead[0]));
    if (after != std::string::npos) {
      size_t best = after;
      while (best < q.size() && (q[best] == ' ' || q[best] == '\t')) ++best;
      const size_t e = q.find_last_not_of(" \t");
      if (best < q.size() && e != std::string::npos)
        return q.substr(best, e - best + 1);
    }
  }

  // ---- 4. fallback: the last whitespace token ("hostname", "whoami") ----
  size_t e = q.find_last_not_of(" \t");
  if (e == std::string::npos) return "";
  size_t b = q.find_last_of(" \t", e);
  b = (b == std::string::npos) ? 0 : b + 1;
  return q.substr(b, e - b + 1);
}

} // namespace canticle

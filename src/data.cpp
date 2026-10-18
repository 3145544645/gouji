// ============================================================================
//  Synthetic instruction data for the tool-use protocol.
//
//  WHAT CHANGED, AND WHY IT MATTERS
//  --------------------------------
//  The first version emitted exactly ONE template per tool:
//      read_file  ->  "read the file X"
//      list_dir   ->  "list the directory X"
//  and so on. The memory therefore contained no pairing between "show me X" and
//  read_file, because such a pairing had never existed anywhere in the corpus.
//  A semantic-robustness probe scored 63.6% and the remaining failures were not
//  a reasoning gap -- the model had simply never seen a second way to say it.
//
//  Paraphrase cannot be inferred from a corpus that contains none. This version
//  emits MANY surface forms per intent, which is the only place synonymy can come
//  from in a system trained on its own generated data.
//
//  Diversity is deliberate and structured:
//    * several verb families per tool
//    * optional politeness prefixes and connectives
//    * Chinese forms alongside English
//    * varying path shapes (drive roots, extensions, bare names)
//
//  Index safety: each array is indexed by its OWN distribution whose bound
//  matches that array. A shared pick(0,3) over a 3-element array once caused an
//  access violation before the first printf, i.e. a silent exit with no output.
// ============================================================================
#include "canticle/data.h"
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <algorithm>
#include <cstring>

namespace canticle {

// ---------------------------------------------------------------------------
//  Surface-form banks. Every entry maps to the SAME tool, so co-occurrence
//  across entries is what teaches the encoder that they mean the same thing.
// ---------------------------------------------------------------------------
static const char* kReadVerb[] = {
  "read the file ", "show me ", "open ", "display ", "print ", "cat ",
  "view ", "fetch ", "load ", "inspect ", "examine ", "peek at ",
  "give me ", "let me see ", "show the contents of ", "output ",
};
static const char* kListVerb[] = {
  "list the directory ", "ls ", "dir ", "list ", "enumerate ", "show the entries of ",
  "what is in ", "what's in ", "list files in ", "show files in ", "contents of ",
  "enumerate files under ",
};
static const char* kWriteVerb[] = {
  "write hello to ", "save hello into ", "create ", "put hello in ", "store hello at ",
  "write to ", "save to ", "make a file ", "dump hello to ",
};
static const char* kShellVerb[] = {
  "run the command ", "execute ", "call ", "please run ", "invoke ",
  "run ", "launch ", "start ",
};
static const char* kCalcVerb[] = {
  "what is ", "compute ", "calculate ", "how much is ", "evaluate ",
  "work out ", "solve ", "what does ", "tell me ",
};
// Polite prefixes may precede any of the above.
static const char* kPrefix[] = { "", "", "", "please ", "can you ", "could you ", "kindly ", "hey " };
// Chinese forms, paired with the same tools.
static const char* kReadZh[]  = { "读取 ", "查看 ", "打印 ", "显示 ", "打开 " };
static const char* kListZh[]  = { "列出 ", "列举 ", "显示目录 ", "看看 " };
static const char* kWriteZh[] = { "写入 hello 到 ", "保存 hello 到 ", "创建 " };
static const char* kShellZh[] = { "执行 ", "运行 ", "调用 " };
static const char* kCalcZh[]  = { "计算 ", "算一下 ", "求 " };

static const char* pick(const char** arr, int n, std::mt19937& r) {
  return arr[r() % (size_t)n];
}

namespace detail {
std::unordered_map<std::string, int>& last_tool_coverage() {
  static std::unordered_map<std::string, int> m;
  return m;
}
}  // namespace detail

std::vector<Sample> synth_tool_data(int n, uint64_t seed) {
  std::mt19937 rng((uint32_t)seed);
  std::vector<Sample> out;
  out.reserve((size_t)n + n / 8 + 8);

  const char* names[] = {"alpha", "beta", "gamma", "delta", "notes", "todo", "log", "cfg"};
  const char* exts[]  = {".txt", ".md", ".json", ".log"};
  const char* roots[] = {"C:/work", "C:/tmp", "D:/data", "C:/log"};
  const char* cmds[]  = {"Get-Date", "hostname", "Get-Location", "whoami"};
  const char* dirs[]  = {"C:/", "C:/Users", "C:/tmp", "D:/"};

  std::uniform_int_distribution<int> pickRoot(0, 3), pickExt(0, 3);
  std::uniform_int_distribution<int> pickCmd(0, 3), pickDir(0, 3);
  std::uniform_int_distribution<int> pickName(0, 3);
  std::uniform_int_distribution<int> pickPre(0, (int)(sizeof(kPrefix) / sizeof(kPrefix[0])) - 1);
  std::uniform_int_distribution<int> coin(0, 1);

  auto path = [&](bool odd) {
    std::string s = roots[pickRoot(rng)];
    s += "/";
    s += names[pickName(rng) * 2 + (odd ? 1 : 0)];
    s += exts[pickExt(rng)];
    return s;
  };
  auto pre = [&]() { return std::string(pick(kPrefix, (int)(sizeof(kPrefix) / sizeof(kPrefix[0])), rng)); };

  // Surface forms for the two read-only additions.
  //
  // Without entries here the tools exist and are unreachable: retrieval can only
  // select from what is stored, so a capability with no corpus is a capability the
  // system will never choose. The generator and the recogniser are two halves of
  // one contract (defect 17), and this is the generator half.
  static const char* kStatVerb[] = {
    "how big is ", "how large is ", "what is the size of ", "size of ",
    "how many bytes is ", "when was ", "when was the last time ",
    "what is the modification time of ", "tell me the size of ",
  };
  static const char* kSearchVerb[] = {
    "find all ", "find any ", "find the ", "search for ", "search in ",
    "grep for ", "look for ", "locate ", "show me files containing ",
  };
  std::uniform_int_distribution<int> pickStat(0, (int)(sizeof(kStatVerb)/sizeof(kStatVerb[0])) - 1);
  std::uniform_int_distribution<int> pickSearch(0, (int)(sizeof(kSearchVerb)/sizeof(kSearchVerb[0])) - 1);
  const char* needles[] = {"TODO", "FIXME", "main", "config"};

  // ---------------------------------------------------------------------------
  //  HOW EACH TOOL IS SCHEDULED, and the property this schedule must have.
  //
  //  The five original tools share i % 5. Every other tool is claimed by a
  //  separate modulus earlier in the loop, which means:
  //
  //    * the moduli are NOT mutually exclusive with %5 -- i=6 satisfies both
  //      "i % 7 == 6" (search) and "i % 5 == 1" (write_file), and search wins
  //      because its test comes first. The overlapping iterations are therefore
  //      not write_file samples at all, so the five original tools receive fewer
  //      than n/5 each.
  //
  //    * the sparse tools get samples at n/7, n/11 and n/13. At n=100 that is 14
  //      stat, 9 mkdir and 7 copy; at n=50 it is 7, 4 and 3.
  //
  //  That sparsity is the root of defect 67: the precedence hunt took "the first
  //  six samples of each tool" from a corpus whose newer tools only appear every
  //  11th or 13th iteration, collected ZERO of them for four tools, and reported a
  //  confident "unknown" baseline built from an empty bucket.
  //
  //  The schedule is left as it is, because the stored 13500-episode memory was
  //  built with it and changing the mix would move every measured number. What is
  //  added instead is a REPORT: the generator counts what it produced per tool, and
  //  a caller can ask. A sparse tool is then visible at generation time rather than
  //  twenty steps later as a mysterious empty result.
  //
  //  Making a hidden condition observable is the same remedy applied to defect 49
  //  (the bucket count) and 75 (the bucket mapping): the schedule is not wrong, its
  //  effects were just not stated anywhere.
  // ---------------------------------------------------------------------------
  // Per-tool tally, so sparsity is observable. See the note above.
  std::unordered_map<std::string, int> produced;
  auto tally = [&](const std::string& reply) {
    const size_t a = reply.find("<|tool|>");
    if (a == std::string::npos) return;
    size_t b = reply.find(' ', a);
    produced[reply.substr(a + 8, (b == std::string::npos ? reply.size() : b) - (a + 8))]++;
  };

  for (int i = 0; i < n; ++i) {
    // One turn in seven teaches a stat or search, interleaved so both appear
    // throughout the shuffled corpus rather than in one contiguous run.
    if (i % 7 == 5) {
      const std::string p = path(false);
      const std::string u = pre() + pick(kStatVerb, (int)(sizeof(kStatVerb)/sizeof(kStatVerb[0])), rng) + p;
      out.push_back({u, "<|tool|>stat {\"path\":\"" + p + "\"}<|/tool|>"});
      continue;
    }
    if (i % 13 == 11) {
      // copy: BOTH paths come from the query, so the stored template is a skeleton
      // whose src/dst are overwritten at retrieval time. The corpus still needs
      // entries so the form is represented in the key space.
      const std::string a = path(false);
      const std::string b = path(true);
      static const char* kCopyVerb[] = {"copy ", "please copy ", "duplicate "};
      const std::string u = pre() + pick(kCopyVerb, 3, rng) + a + " to " + b;
      out.push_back({u, "<|tool|>copy {\"src\":\"" + a + "\",\"dst\":\"" + b + "\"}<|/tool|>"});
      continue;
    }
    if (i % 11 == 9) {
      // mkdir: the path is the payload, as for read_file and stat, so the existing
      // single-payload transfer moves the right thing without special handling.
      const std::string d = dirs[pickDir(rng)];
      const std::string nm = names[pickName(rng)];
      const std::string u = pre() + std::string("make a directory ") + d + "/" + nm;
      out.push_back({u, "<|tool|>mkdir {\"path\":\"" + d + "/" + nm + "\"}<|/tool|>"});
      continue;
    }
    if (i % 7 == 6) {
      // The PATH is part of the prompt, and it comes LAST.
      //
      // The first version omitted it -- "find all main" -> search {"path":"D:/",
      // "pattern":"main"} -- which made the stored payload the NEEDLE while a real
      // user's payload is the PATH. Slot transfer then substituted the path into
      // the pattern field, producing
      //     search {"path":"D:/","pattern":"C:/Users/cc/canticle/src"}
      // i.e. searching D:/ for a file whose "content" is a path.
      //
      // search is the first tool with TWO arguments, and slot transfer moves ONE
      // payload. Aligning the corpus so that the payload it moves is the path (as
      // for stat and read_file) leaves the pattern to be filled from the query by
      // a dedicated extractor -- see extract_search_pattern.
      const std::string d = dirs[pickDir(rng)];
      const std::string needle = needles[rng() % 4];
      const std::string u = pre() + pick(kSearchVerb, (int)(sizeof(kSearchVerb)/sizeof(kSearchVerb[0])), rng)
                          + needle + " in " + d;
      out.push_back({u, "<|tool|>search {\"path\":\"" + d + "\",\"pattern\":\"" + needle + "\"}<|/tool|>"});
      continue;
    }
    switch (i % 5) {
      case 0: {
        const std::string p = path(false);
        std::string u;
        if (coin(rng)) {                       // English
          u = pre() + pick(kReadVerb, (int)(sizeof(kReadVerb)/sizeof(kReadVerb[0])), rng) + p;
        } else {                               // Chinese
          u = pick(kReadZh, (int)(sizeof(kReadZh)/sizeof(kReadZh[0])), rng) + p;
        }
        out.push_back({u, "<|tool|>read_file {\"path\":\"" + p + "\"}<|/tool|>"});
        break;
      }
      case 1: {
        const std::string p = path(true);
        std::string u;
        if (coin(rng)) {
          u = pre() + pick(kWriteVerb, (int)(sizeof(kWriteVerb)/sizeof(kWriteVerb[0])), rng) + p;
        } else {
          u = pick(kWriteZh, (int)(sizeof(kWriteZh)/sizeof(kWriteZh[0])), rng) + p;
        }
        out.push_back({u, "<|tool|>write_file {\"path\":\"" + p + "\",\"content\":\"hello\"}<|/tool|>"});
        break;
      }
      case 2: {
        const std::string p = dirs[pickDir(rng)];
        std::string u;
        if (coin(rng)) {
          u = pre() + pick(kListVerb, (int)(sizeof(kListVerb)/sizeof(kListVerb[0])), rng) + p;
        } else {
          u = pick(kListZh, (int)(sizeof(kListZh)/sizeof(kListZh[0])), rng) + p;
        }
        out.push_back({u, "<|tool|>list_dir {\"path\":\"" + p + "\"}<|/tool|>"});
        break;
      }
      case 3: {
        const std::string c = cmds[pickCmd(rng)];
        std::string u;
        if (coin(rng)) {
          u = pre() + pick(kShellVerb, (int)(sizeof(kShellVerb)/sizeof(kShellVerb[0])), rng) + c;
        } else {
          u = pick(kShellZh, (int)(sizeof(kShellZh)/sizeof(kShellZh[0])), rng) + c;
        }
        out.push_back({u, "<|tool|>run_shell {\"cmd\":\"" + c + "\"}<|/tool|>"});
        break;
      }
      default: {
        const int a = (int)(rng() % 90) + 10, b = (int)(rng() % 90) + 10;
        const char op = "+-*"[rng() % 3];
        const std::string expr = std::to_string(a) + std::string(1, op) + std::to_string(b);
        const std::string pretty = std::to_string(a) + " " + std::string(1, op) + " " + std::to_string(b);
        std::string u;
        if (coin(rng)) {
          u = pre() + pick(kCalcVerb, (int)(sizeof(kCalcVerb)/sizeof(kCalcVerb[0])), rng) + pretty;
        } else {
          u = pick(kCalcZh, (int)(sizeof(kCalcZh)/sizeof(kCalcZh[0])), rng) + pretty;
        }
        out.push_back({u, "<|tool|>calc {\"expr\":\"" + expr + "\"}<|/tool|>"});
        break;
      }
    }
  }

  // A slice of plain chat so the model does not answer EVERY prompt with a tool.
  const char* cq[] = {"who are you", "hello", "what can you do", "hi there"};
  const char* ca[] = {
    "I am CANTICLE, a tiny local model running on this GPU.",
    "Hello. I am ready.",
    "I read and write files, list directories, run commands and do arithmetic.",
    "Hi. What do you need?"};
  for (int i = 0; i < n / 8; ++i) {
    const int k = (int)(rng() % 4);
    out.push_back({cq[k], ca[k]});
  }

  std::shuffle(out.begin(), out.end(), rng);

  // Tally AFTER the fact rather than at every push_back: one pass over the result
  // cannot miss a branch, whereas a per-branch call can -- and a tally that quietly
  // undercounts would understate exactly the sparsity it exists to reveal.
  for (const Sample& s : out) tally(s.reply);
  detail::last_tool_coverage() = produced;

  // Report only when asked -- a generator that printed on every call would bury
  // real output, and this is diagnostic rather than per-run information.
  if (std::getenv("AM_DATA_COVERAGE")) {
    std::fprintf(stderr, "[data] %d samples, per-tool counts:\n", (int)out.size());
    for (const auto& kv : produced)
      std::fprintf(stderr, "  %-11s %5d  (%.1f%%)\n", kv.first.c_str(), kv.second,
                   100.0 * kv.second / (double)(out.size() ? out.size() : 1));
    std::fprintf(stderr, "  A count of 0, or an unexpectedly small count, means every\n"
                         "  test that samples this tool measures less than it appears to.\n");
  }
  return out;
}

} // namespace canticle

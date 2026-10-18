
// ============================================================================
//  CANTICLE chat + tool-use layer.
//
//  The model is 25M parameters at 2 bits per weight, so it is not going to
//  reason like a frontier model. What it CAN do reliably is follow a rigid
//  protocol. Every tool call is therefore emitted in an unambiguous,
//  single-line, parseable form rather than as free-form JSON:
//
//      <|tool|>name {"key":"value"}<|/tool|>
//
//  The parser is deliberately forgiving (finds the markers anywhere in the
//  stream) because a 25M model will sometimes wrap them in prose.
// ============================================================================
#include "canticle/model.h"
#include <sstream>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <chrono>
#include <filesystem>
#include <cstdlib>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

namespace canticle {

// NOTE: json_get_string / json_get_number live in tokenizer.cpp (one
// definition per symbol). Do not re-implement them here.

static std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    switch (c) {
      case '"':  o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n";  break;
      case '\r': o += "\\r";  break;
      case '\t': o += "\\t";  break;
      default:   o.push_back(c); break;
    }
  }
  return o;
}

// ----------------------------------------------------------------- tools
const std::vector<ToolSpec>& tool_specs() {
  static const std::vector<ToolSpec> specs = {
    { "read_file",  "read a UTF-8 text file",
      "<|tool|>read_file {\"path\":\"C:\\\\tmp\\\\a.txt\"}<|/tool|>", tool_read_file , "read the file " , 1, "read the file C:/tmp/a.txt" , Effect::Pure },
    { "write_file", "create or overwrite a UTF-8 text file",
      "<|tool|>write_file {\"path\":\"C:\\\\tmp\\\\a.txt\",\"content\":\"hi\"}<|/tool|>", tool_write_file , "write hello to " , 3, "write hello to C:/tmp/a.txt" , Effect::Irreversible },
    { "list_dir",   "list entries of a directory",
      "<|tool|>list_dir {\"path\":\"C:\\\\tmp\"}<|/tool|>", tool_list_dir , "list the directory " , 2, "list the directory C:/tmp" , Effect::Pure },
    { "run_shell",  "run a PowerShell command and capture output",
      "<|tool|>run_shell {\"cmd\":\"Get-Date\"}<|/tool|>", tool_run_shell , "run the command " , 4, "run the command hostname" , Effect::Irreversible },
    { "calc",       "evaluate a simple arithmetic expression",
      "<|tool|>calc {\"expr\":\"12*(3+4)\"}<|/tool|>", tool_calc , "what is " , 5, "what is 40 - 14" , Effect::Pure },
    { "stat",       "size, type and modification time of one path",
      "<|tool|>stat {\"path\":\"C:\\tmp\\a.txt\"}<|/tool|>", tool_stat , "how big is " , 6, "how big is C:/tmp/a.txt" , Effect::Pure },
    { "search",     "find files by name or content under a directory",
      "<|tool|>search {\"path\":\"C:\\tmp\",\"pattern\":\"TODO\"}<|/tool|>", tool_search , nullptr , 7, "find all TODO in C:/tmp" , Effect::Pure },
    { "mkdir",      "create a directory (and any missing parents)",
      "<|tool|>mkdir {\"path\":\"C:\\tmp\\newdir\"}<|/tool|>", tool_mkdir , "make a directory " , 8, "make a directory C:/tmp/nd" , Effect::Compensable },
    { "copy",       "copy a file or directory (does not overwrite)",
      "<|tool|>copy {\"src\":\"C:\\tmp\\a.txt\",\"dst\":\"C:\\tmp\\b.txt\"}<|/tool|>", tool_copy , nullptr , 9, "copy C:/tmp/a.txt to C:/tmp/b.txt" , Effect::Compensable },
  };
  return specs;
}

// ------------------------------------------------------------------ render
std::string render_chat(const std::vector<std::pair<std::string,std::string>>& turns,
                        const std::string& system_prompt) {
  // Template MUST match pack_samples() byte for byte. Training emits the
  // markers with NO surrounding newlines:
  //   <|system|>\n<sys>\n<|user|>prompt<|assistant|>reply
  // A stray newline here puts the model off-distribution at inference time.
  std::string s;
  s += "<|system|>\n" + system_prompt + "\n";
  for (auto& t : turns) {
    if (t.first == "user")           s += "<|user|>" + t.second;
    else if (t.first == "assistant") s += "<|assistant|>" + t.second;
    else if (t.first == "tool")      s += "<|tool_out|>" + t.second;
  }
  s += "<|assistant|>";
  return s;
}

std::vector<int> render_chat_ids(const std::vector<std::pair<std::string,std::string>>& turns,
                                 const std::string& system_prompt,
                                 const BPETokenizer& tk) {
  std::vector<int> ids;
  auto push = [&](const char* sp) {
    auto it = tk.enc.find(sp);
    if (it != tk.enc.end()) ids.push_back(it->second);
  };
  auto append = [&](const std::string& s) {
    for (int id : tk.encode(s)) ids.push_back(id);
  };

  // header: system marker + newline + system text + newline, encoded as ONE
  // chunk, exactly as pack_samples_impl does it.
  append(std::string("<|system|>\n") + system_prompt + "\n");

  for (auto& t : turns) {
    if (t.first == "user") {
      push("<|user|>");          // its own token id, never merged
      append(t.second);
    } else if (t.first == "assistant") {
      push("<|assistant|>");
      append(t.second);
    } else if (t.first == "tool") {
      push("<|tool_out|>");
      append(t.second);
    }
  }
  push("<|assistant|>");
  return ids;
}

// ------------------------------------------------------------------- parse
std::vector<ToolCall> parse_tool_calls(const std::string& text) {
  std::vector<ToolCall> out;
  const std::string open = "<|tool|>", close = "<|/tool|>";
  size_t p = 0;
  while (true) {
    size_t a = text.find(open, p);
    if (a == std::string::npos) break;
    size_t b = text.find(close, a + open.size());
    if (b == std::string::npos) break;
    std::string body = text.substr(a + open.size(), b - a - open.size());
    // trim
    size_t s0 = body.find_first_not_of(" \t\r\n");
    size_t s1 = body.find_last_not_of(" \t\r\n");
    if (s0 != std::string::npos) {
      body = body.substr(s0, s1 - s0 + 1);
      size_t sp = body.find_first_of(" \t{");
      ToolCall c;
      c.name = body.substr(0, sp);
      size_t jb = body.find('{', sp);
      c.args_json = (jb == std::string::npos) ? "{}" : body.substr(jb);
      // strip a trailing closing brace run that belongs to nothing else
      if (!c.name.empty()) out.push_back(c);
    }
    p = b + close.size();
  }
  return out;
}

std::string strip_tool_calls(const std::string& text) {
  std::string s = text;
  const std::string open = "<|tool|>", close = "<|/tool|>";
  size_t p = 0;
  while (true) {
    size_t a = s.find(open, p);
    if (a == std::string::npos) break;
    size_t b = s.find(close, a);
    if (b == std::string::npos) break;
    s.erase(a, b + close.size() - a);
    p = a;
  }
  return s;
}

// -------------------------------------------------------------- execution
std::string tool_read_file(const std::string& args) {
  std::string path = json_get_string(args, "path");
  if (path.empty()) return "error: missing \"path\"";
  std::ifstream f(path, std::ios::binary);
  if (!f) return "error: cannot open " + path;
  std::stringstream ss; ss << f.rdbuf();
  std::string s = ss.str();
  // 4000 bytes was chosen for a single-shot tool call. In a conversation that
  // floods the terminal on every turn, so the transcript is capped more
  // tightly; the tool still reports the true size so nothing is hidden.
  const size_t LIMIT = 600;
  if (s.size() > LIMIT)
    s = s.substr(0, LIMIT) + "\n...[truncated, file is " +
        std::to_string(s.size()) + " bytes]";
  return s;
}

std::string tool_write_file(const std::string& args) {
  std::string path = json_get_string(args, "path");
  std::string content = json_get_string(args, "content");
  if (path.empty()) return "error: missing \"path\"";
  // Create the parent directory rather than failing on a missing one: a model
  // that correctly constructs "C:/new/dir/file.txt" should not be reported as
  // wrong just because the directory does not exist yet.
  std::error_code ec;
  std::filesystem::path p(path);
  if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);

  // Write through the Win32 API rather than std::ofstream.
  //
  // On Windows, ofstream with a forward-slash path ("C:/tmp/x.md") failed with
  // a bare "cannot write" while the directory provably existed and the same
  // path worked from PowerShell. CreateFileA with the normalised separator set
  // is unambiguous, and it also reports the actual error code instead of
  // leaving us guessing.
  std::string native = p.string();
  for (char& c : native) if (c == '/') c = '\\';

  // ---------------------------------------------------------------------------
  //  CREATE_NEW, NOT CREATE_ALWAYS.
  //
  //  Measured: "create a summary of <existing file>" routed to write_file and
  //  REPLACED the file with five bytes of "hello". The original content was gone and
  //  nothing had reported a problem. The audit that found this is the reason the
  //  tool is described as "create or overwrite" -- and the description was the bug:
  //  a router that guesses its destination must never also be willing to destroy
  //  what is there.
  //
  //  copy already refuses to overwrite, with a comment explaining that a misread
  //  destination must not be able to lose data. The same rule applies here and was
  //  simply not applied. CREATE_NEW enforces it at the OS level rather than by a
  //  pre-check that could race.
  //
  //  An overwrite remains possible through an explicit flag, so the capability is
  //  not lost -- only the SILENT version of it.
  // ---------------------------------------------------------------------------
  const bool allow_overwrite = (args.find("\"overwrite\":true") != std::string::npos);
  HANDLE h = CreateFileA(native.c_str(), GENERIC_WRITE, 0, nullptr,
                         allow_overwrite ? CREATE_ALWAYS : CREATE_NEW,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_EXISTS) {
    return "error: " + p.string() + " already exists. Refusing to overwrite it; "
           "delete it first, or say \"overwrite\" explicitly.";
  }
  if (h == INVALID_HANDLE_VALUE) {
    const DWORD e = GetLastError();
    // ERROR_ACCESS_DENIED (5) on Windows means the OS refused, which in a
    // sandboxed session usually means the path lies outside the writable
    // root rather than that the caller is buggy. Say so, instead of emitting a
    // bare "cannot write" that sends the next reader hunting for a code fault.
    // Verified boundary: writing under the workspace succeeds, writing to
    // C:\tmp or %TEMP% returns err 5.
    std::string why = (e == 5)
      ? " (access denied -- this path is outside the writable root)"
      : (" (Win32 err " + std::to_string(e) + ")");
    return "error: cannot write " + path + why + ", resolved=" + native;
  }
  DWORD wrote = 0;
  BOOL ok = WriteFile(h, content.data(), (DWORD)content.size(), &wrote, nullptr);
  if (!ok && !content.empty()) {
    DWORD e = GetLastError();
    CloseHandle(h);
    return "error: WriteFile failed (Win32 err " + std::to_string(e) + ")";
  }
  CloseHandle(h);
  return "wrote " + std::to_string(wrote) + " bytes to " + path;
}

std::string tool_list_dir(const std::string& args) {
  std::string path = json_get_string(args, "path");
  if (path.empty()) path = ".";
  std::string out;
  int shown = 0;
  try {
    for (auto& e : std::filesystem::directory_iterator(path)) {
      out += e.is_directory() ? "[D] " : "[F] ";
      out += e.path().filename().string();
      if (e.is_regular_file()) out += "  (" + std::to_string(e.file_size()) + " B)";
      out += "\n";
      if (++shown >= 60) { out += "...[truncated]\n"; break; }
    }
  } catch (const std::exception& e) {
    return std::string("error: ") + e.what();
  }
  return out.empty() ? "(empty directory)" : out;
}

std::string tool_run_shell(const std::string& args) {
  std::string cmd = json_get_string(args, "cmd");
  if (cmd.empty()) return "error: missing \"cmd\"";

  // Redirect through a temp file. Passing the command straight to std::system
  // with a "pwsh -Command \"...\"" wrapper broke on every command containing a
  // quote or a pipe, and the failure was silent (exit=1, no output).
  // ---------------------------------------------------------------------------
  //  TWO BUGS, both found by an independent audit, both silent.
  //
  //  1. THE OUTPUT WAS ALWAYS EMPTY. The command redirected into
  //     fs::temp_directory_path(), and the read-back returned nothing on this
  //     machine -- so "run the command hostname" reported "(no output)" while the
  //     same redirection run by hand produced the hostname. A tool whose entire
  //     purpose is to report what a command did reported nothing.
  //
  //     Fixed by capturing through a pipe instead of a file: no temp path, no
  //     dependency on which directories are writable, no cleanup to get wrong.
  //
  //  2. THE EXIT CODE WAS ALWAYS 0. "rc >> 8" is the Unix convention, where system()
  //     returns a wait status. On Windows it returns the exit code itself, so
  //     shifting turned every code below 256 into 0. Measured: "cmd /c exit 3" was
  //     reported as exit=0.
  //
  //     That in turn made the nonzero-exit classification in dispatch_tool_result
  //     unreachable, and its comment -- "a non-zero code is a real failure that the
  //     caller previously could not see" -- described a capability that did not
  //     exist.
  //
  //  Both are the same kind of defect as the vacuous tests: a mechanism that looks
  //  present, reports success, and does nothing.
  // ---------------------------------------------------------------------------
  const std::string full = "cmd /c " + cmd;
  std::string s;
  int rc = -1;
  {
    FILE* pipe = _popen(full.c_str(), "rb");
    if (!pipe) return "error: cannot spawn a shell";
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), pipe)) > 0) s.append(buf, n);
    rc = _pclose(pipe);   // Windows: the exit code directly, no shift
  }

  const size_t LIMIT = 4000;
  if (s.size() > LIMIT) s = s.substr(0, LIMIT) + "\n...[truncated]";
  if (s.empty()) s = "(no output)";
  return "exit=" + std::to_string(rc) + "\n" + s;
}

// ------------------------------------------------------------------ stat
//
// Read-only metadata. Implemented because "how big is X" and "when was X
// modified" were previously classified as Unsupported -- correct at the time,
// since no tool answered them -- and routing them to read_file would have opened
// the file, which is not what was asked.
//
// Deliberately read-only: no timestamps are set, no permissions are changed.
// Anything that modifies metadata stays in the unsupported list.
std::string tool_stat(const std::string& args) {
  std::string path = json_get_string(args, "path");
  if (path.empty()) return "error: missing \"path\"";
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::exists(path, ec)) return "error: cannot find " + path;

  std::string out;
  const bool isdir = fs::is_directory(path, ec);
  out += std::string("type      : ") + (isdir ? "directory" : "file") + "\n";
  if (!isdir) {
    const auto sz = fs::file_size(path, ec);
    if (!ec) out += "size      : " + std::to_string(sz) + " bytes\n";
  }

  // Modification time. std::filesystem reports a file_clock value; converting via
  // system_clock is the portable route and avoids a platform-specific API.
  const auto mt = fs::last_write_time(path, ec);
  if (!ec) {
    const auto sys = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        mt - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
    const std::time_t t = std::chrono::system_clock::to_time_t(sys);
    char buf[64] = {0};
    std::tm tmv{};
#if defined(_WIN32)
    if (localtime_s(&tmv, &t) == 0)
      std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
#else
    if (std::localtime_r(&t, &tmv))
      std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
#endif
    out += std::string("modified  : ") + buf + "\n";
  }

  if (isdir) {
    int n = 0;
    for (auto it = fs::directory_iterator(path, ec); !ec && it != fs::directory_iterator(); ++it) ++n;
    out += "entries   : " + std::to_string(n) + "\n";
  }
  return out;
}

// ------------------------------------------------------------------- copy
//
// Copies a file or directory to a new path. ADDITIVE: the source is only read, so
// nothing that existed before can be lost. That is the property that separates it
// from delete/rename/move, which remain unimplemented.
//
// Refuses to overwrite an existing destination. silent replacement would make a
// misread destination destroy data, which is exactly the risk this tool is
// supposed to avoid -- and the caller can always delete first if that is what they
// meant, as an explicit separate act.
std::string tool_copy(const std::string& args) {
  namespace fs = std::filesystem;
  const std::string src = json_get_string(args, "src");
  const std::string dst = json_get_string(args, "dst");
  if (src.empty()) return "error: missing \"src\"";
  if (dst.empty()) return "error: missing \"dst\"";

  std::error_code ec;
  if (!fs::exists(src, ec)) return "error: cannot find " + src;
  if (fs::exists(dst, ec)) return "error: destination already exists: " + dst;

  // "copy X to C:/dir/" means "into that directory, keeping the name"
  std::string target = dst;
  if (dst.back() == '/' || dst.back() == '\\') target = dst + fs::path(src).filename().string();

  // ---------------------------------------------------------------------------
  //  A FAILED COPY MUST NOT LEAVE A PARTIAL TREE.
  //
  //  Measured, before this wrapper:
  //
  //      src/ contains a.txt, b.txt, and c.txt (c.txt held open by another
  //      process). "copy src to dst" returned
  //          error: cannot copy directory: ... being used by another process
  //      and left dst/a.txt and dst/b.txt behind.
  //
  //  The message was honest and the outcome was not: the caller is told the copy
  //  failed, and a half-populated directory is sitting there. A later retry then
  //  fails with "destination already exists", so the failure is not even
  //  recoverable by repeating the request. ROADMAP stage 3 names this explicitly
  //  as its hard part -- "a mid-step failure leaves dirty state" -- and it was
  //  already true of a single-step command.
  //
  //  So: copy into a temporary sibling, and rename it into place only on success.
  //  The rename is the commit point. On any failure the temporary tree is removed,
  //  and the destination either does not exist or is the old one, unchanged.
  //
  //  The temporary name is derived from the target so two concurrent copies of
  //  different sources do not collide, and it ends in a suffix no ordinary path
  //  would use, so a crash mid-copy leaves something obviously discardable.
  // ---------------------------------------------------------------------------
  const bool is_dir = fs::is_directory(src, ec);
  const std::string staging = target + ".canticle-partial";
  std::error_code rm_ec;
  fs::remove_all(staging, rm_ec);   // a leftover from a previous crash

  auto fail = [&](const std::string& what) {
    std::error_code c2;
    fs::remove_all(staging, c2);    // roll back; ignore failure, we are failing
    return "error: cannot " + what + ": " + ec.message();
  };

  if (is_dir) {
    fs::copy(src, staging, fs::copy_options::recursive, ec);
    if (ec) return fail("copy directory");
    fs::rename(staging, target, ec);
    if (ec) return fail("move directory into place");
    return "copied directory " + src + " to " + target;
  }

  fs::copy_file(src, staging, ec);
  if (ec) return fail("copy file");
  fs::rename(staging, target, ec);
  if (ec) return fail("move file into place");
  const auto sz = fs::file_size(target, ec);
  return "copied " + src + " to " + target + " (" + std::to_string(ec ? 0 : sz) + " bytes)";
}

// ------------------------------------------------------------------ mkdir
//
// Creates a directory. Included in the read-only round because creating is
// ADDITIVE: it cannot destroy anything that already exists, and if the path is
// occupied the call fails rather than overwriting.
//
// destroy is deliberately NOT implemented. delete, rename and move all remove
// data, and an irreversible operation performed on a request the model may have
// misinterpreted is a different class of risk from an unwanted read. Those need a
// confirmation step, which is a design decision rather than a tool addition.
std::string tool_mkdir(const std::string& args) {
  namespace fs = std::filesystem;
  std::string path = json_get_string(args, "path");
  if (path.empty()) return "error: missing \"path\"";
  std::error_code ec;
  if (fs::exists(path, ec)) {
    if (fs::is_directory(path, ec)) return "already exists: " + path;
    return "error: a file with that name already exists: " + path;
  }
  // create_directories makes intermediate levels, so "C:/a/b/c" works even when
  // none of a/b/c exist.
  if (!fs::create_directories(path, ec) || ec)
    return "error: cannot create " + path + (ec ? (" (" + ec.message() + ")") : "");
  return "created directory " + path;
}

// ---------------------------------------------------------------- search
//
// Read-only name and content search.
//
// Bounded on purpose: a recursive walk of a large tree can run for minutes and
// produce a megabyte of matches, neither of which belongs in a tool result. The
// caps are reported rather than applied silently, so a truncated answer is never
// mistaken for a complete one.
//
// Nothing here writes, moves or removes anything: a match is only ever reported.
std::string tool_search(const std::string& args) {
  namespace fs = std::filesystem;
  std::string root = json_get_string(args, "path");
  std::string pat  = json_get_string(args, "pattern");
  if (root.empty()) root = ".";
  if (pat.empty()) return "error: missing \"pattern\"";

  const int MAX_MATCHES = 40;
  const int MAX_FILES   = 2000;   // how many files may be opened
  const uintmax_t MAX_BYTES = 4u << 20;  // skip files larger than 4 MiB

  std::string out;
  int found = 0, scanned = 0;
  bool capped = false;

  std::error_code ec;
  if (!fs::exists(root, ec)) return "error: cannot find " + root;

  auto report = [&](const std::string& p, const std::string& why) {
    out += p + "  " + why + "\n";
    ++found;
  };

  for (auto it = fs::recursive_directory_iterator(
           root, fs::directory_options::skip_permission_denied, ec);
       !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
    const auto& e = *it;
    const std::string name = e.path().filename().string();
    if (e.is_directory()) {
      if (name.find(pat) != std::string::npos) {
        report(e.path().string(), "[dir name]");
        if (found >= MAX_MATCHES) { capped = true; break; }
      }
      continue;
    }
    if (!e.is_regular_file()) continue;
    if (++scanned > MAX_FILES) { capped = true; break; }

    if (name.find(pat) != std::string::npos) {
      report(e.path().string(), "[filename]");
      if (found >= MAX_MATCHES) { capped = true; break; }
      continue;   // already matched by name; no need to read it
    }
    // content match, for small files only
    const auto sz = fs::file_size(e.path(), ec);
    if (ec || sz > MAX_BYTES) continue;
    std::ifstream f(e.path(), std::ios::binary);
    if (!f) continue;
    std::string line;
    while (std::getline(f, line)) {
      if (line.find(pat) != std::string::npos) {
        std::string trimmed = line.substr(0, 120);
        report(e.path().string(), "[content] " + trimmed);
        break;   // one hit per file is enough to locate it
      }
    }
    if (found >= MAX_MATCHES) { capped = true; break; }
  }

  if (out.empty()) return "no matches for \"" + pat + "\" under " + root;
  if (capped)
    out += "...[stopped at " + std::to_string(found) + " matches after scanning " +
           std::to_string(scanned) + " files; narrow the path or pattern]\n";
  return out;
}

// -------------------------------------------------------------------- calc
// Tiny recursive-descent arithmetic evaluator: + - * / ( ) and unary minus.
namespace {
struct Calc {
  const char* p; bool err = false;
  void ws() { while (*p == ' ' || *p == '\t') ++p; }
  double expr() {
    double v = term();
    for (;;) { ws();
      if (*p == '+') { ++p; v += term(); }
      else if (*p == '-') { ++p; v -= term(); }
      else return v; }
  }
  double term() {
    double v = unary();
    for (;;) { ws();
      if (*p == '*') { ++p; v *= unary(); }
      else if (*p == '/') { ++p; double d = unary(); if (d == 0) { err = true; return 0; } v /= d; }
      else return v; }
  }
  double unary() { ws(); if (*p == '-') { ++p; return -unary(); } if (*p == '+') { ++p; return unary(); } return atom(); }
  double atom() {
    ws();
    if (*p == '(') { ++p; double v = expr(); ws(); if (*p == ')') ++p; else err = true; return v; }
    char* end = nullptr;
    double v = std::strtod(p, &end);
    if (end == p) { err = true; return 0; }
    p = end; return v;
  }
};
}

std::string tool_calc(const std::string& args) {
  std::string e = json_get_string(args, "expr");
  if (e.empty()) return "error: missing \"expr\"";

  // ---------------------------------------------------------------------------
  //  AN EXPRESSION MUST BE AN EXPRESSION, checked before parsing.
  //
  //  The parser is lenient -- it skips what it does not understand -- so a whole
  //  Chinese sentence was accepted as an "expression" and evaluated:
  //
  //      "计算三角形的面积。"  ->  calc {"expr":"计算三角形的面积。"}  -> a number
  //
  //  Measured on 48818 real Chinese instructions: ten of the first 160 requests
  //  acted on were of this kind. Each produced a confident numeric answer to a
  //  question that was never arithmetic, which is worse than a refusal because the
  //  output looks like a result.
  //
  //  The lenient parser is not the bug -- it exists so partial input can be
  //  diagnosed. The bug is that nothing checked the INPUT was arithmetic before
  //  handing it over. A character whitelist is the narrowest correct test: an
  //  expression contains digits, operators, parentheses, a decimal point, spaces,
  //  and the names of the few functions the evaluator knows.
  //
  //  Anything else -- a letter sequence that is not a known function, or a CJK
  //  character -- means this is not an arithmetic request, and the honest answer is
  //  to say so rather than to evaluate whatever numbers appear inside it.
  // ---------------------------------------------------------------------------
  {
    static const char* kFns[] = {"sqrt", "sin", "cos", "tan", "log", "abs", "pow", "exp"};
    std::string probe = e;
    for (char& c : probe) c = (char)std::tolower((unsigned char)c);
    for (const char* f : kFns) {
      const size_t n = std::strlen(f);
      size_t p = 0;
      while ((p = probe.find(f, p)) != std::string::npos) probe.erase(p, n);
    }
    for (unsigned char c : probe) {
      const bool ok = std::isdigit(c) || c == ' ' || c == '\t' || c == '.' ||
                      c == '+' || c == '-' || c == '*' || c == '/' || c == '%' ||
                      c == '(' || c == ')' || c == ',' || c == '^';
      if (ok) continue;
      return "error: \"" + e + "\" is not an arithmetic expression";
    }
    // and it must contain at least one digit and one operator, or it is a bare
    // number rather than a calculation
    bool digit = false, op = false;
    for (char c : probe) {
      if (std::isdigit((unsigned char)c)) digit = true;
      else if (std::strchr("+-*/%^", c)) op = true;
    }
    if (!digit || !op) return "error: \"" + e + "\" is not an arithmetic expression";
  }

  Calc c; c.p = e.c_str();
  double v = c.expr();
  if (c.err) return "error: cannot parse \"" + e + "\"";
  std::ostringstream os; os << e << " = " << v;
  return os.str();
}

// ---------------------------------------------------------------------------
//  Structured dispatch.
//
//  Each tool body is the existing implementation, wrapped so that the failure
//  classification is explicit at the point where the cause is actually known --
//  a missing file is fixable by changing the path, a bad JSON body is not.
//  Guessing this afterwards from the output text is what the old code did and
//  what made automatic correction unsafe to build on.
// ---------------------------------------------------------------------------
ToolResult dispatch_tool_result(const ToolCall& c) {
  ToolResult r;
  r.name = c.name;

  // An unknown tool is a caller bug, not something to learn around.
  //
  // Membership is decided from the same table dispatch uses, so the two cannot
  // disagree. The previous version was a hand-written chain of != comparisons whose
  // own comment warned it had to be kept in step with dispatch -- and a comment is
  // not a mechanism.
  bool known = false;
  for (const ToolSpec& s : tool_specs())
    if (c.name == s.name) { known = true; break; }
  if (!known) {
    r.ok = false;
    r.fixable = false;
    r.error = "unknown-tool";
    r.output = "error: unknown tool \"" + c.name + "\"";
    return r;
  }

  // A structurally invalid argument block cannot be repaired by trial.
  if (c.args_json.empty() || c.args_json.find('{') == std::string::npos) {
    r.ok = false;
    r.fixable = false;
    r.error = "bad-args";
    r.output = "error: missing argument object for " + c.name;
    return r;
  }

  const std::string full = dispatch_tool(c);
  r.output = full;

  if (full.rfind("error:", 0) == 0) {
    r.ok = false;
    // Distinguish "the world did not cooperate" from "the request was malformed".
    // Only the former is worth remembering as a lesson.
    if (full.find("missing") != std::string::npos)
      { r.error = "bad-args"; r.fixable = false; }
    else if (full.find("cannot parse") != std::string::npos)
      { r.error = "bad-args"; r.fixable = false; }
    else
      { r.error = "unavailable"; r.fixable = true; }   // missing file, denied path
    return r;
  }

  // run_shell reports its exit code inline; a non-zero code is a real failure
  // that the caller previously could not see without parsing the body.
  if (c.name == "run_shell" && full.rfind("exit=", 0) == 0) {
    const int code = std::atoi(full.c_str() + 5);
    if (code != 0) { r.ok = false; r.fixable = true; r.error = "nonzero-exit"; }
    return r;
  }

  // read_file can succeed but return nothing; that is still a success.
  return r;
}

// A lookup, not a chain.
//
// The chain this replaces listed every tool by name, in parallel with the
// whitelist above and with the prompt table. Three lists, three chances to forget
// one -- and defect 70 was exactly a forgotten list. There is now one table, and
// this is a read of it.
std::string dispatch_tool(const ToolCall& c) {
  for (const ToolSpec& s : tool_specs())
    if (c.name == s.name && s.fn) return s.fn(c.args_json);
  return "error: unknown tool \"" + c.name + "\"";
}

// ------------------------------------------------------------------ prompt
// The system prompt is a HARD context budget line. An earlier 700-character
// version tokenised to roughly 400 tokens -- larger than the model's entire
// 256-token window -- so every prompt was truncated down to nothing but the
// preamble and every question produced an identical reply. Keep it terse and
// let the examples in the training data teach the format instead.
// ---------------------------------------------------------------------------
//  DERIVED FROM THE TABLE, like dispatch and the accepted-name check.
//
//  The literal it replaces listed the nine tools by name -- a fourth hand-maintained
//  copy of the same fact, and the one that am-source could not see because that
//  suite never called this function. So the claim in defect 73, "adding a tool is
//  one line", was false: the prompt still needed editing by hand.
//
//  This is the same lesson as defects 73 and 75, arriving one place later. The
//  reason it kept arriving is that "derive it from the table" was applied to the
//  places a grep for the tool names found, and a prompt built by string
//  concatenation does not look like a list until you read it.
// ---------------------------------------------------------------------------
std::string tool_system_prompt() {
  std::string names;
  for (const ToolSpec& s : tool_specs()) {
    if (!names.empty()) names += " ";
    names += s.name;
  }
  return
    "Tools: " + names + ".\n"
    "Use a tool: <|tool|>name {\"key\":\"value\"}<|/tool|>\n"
    "Or reply in one short sentence.\n";
}

} // namespace canticle

#pragma once
#include "canticle/common.h"
#include "canticle/quant.h"
#include <memory>
#include <functional>
#include <random>
#include <unordered_map>

namespace canticle {

// ===========================================================================
//  CANTICLE-1 configuration.
//  Sizing rule, enforced in Model::validate(): the ternary working set
//  (sum of all quantised linear weights) plus the embedding table must fit in
//  the GPU's persisting L2 budget. On a 22 MiB budget that caps us at ~25M
//  parameters. That is the whole thesis: choose the format to fit the cache.
// ===========================================================================
struct Config {
  int vocab     = 8192;
  int d_model   = 512;
  int n_layer   = 8;
  int n_head    = 8;
  int d_ff      = 1536;
  int ctx       = 512;
  float rope_theta = 10000.0f;
  float rms_eps    = 1e-6f;
  int   head_dim() const { return d_model / n_head; }
  int   kv_dim()   const { return d_model; }
};

// A ternary linear layer: weights live as two bit-planes, activations fp16.
struct TernLinear {
  int in = 0, out = 0;
  TernaryMatrix W;                  // [out][in], row-major, in % 32 == 0
  std::vector<uint16_t> scale;      // per-output-channel dequant scale (fp16)
  void alloc(int o, int i) { out = o; in = i; W.alloc(o, i); scale.assign(o, 0); }
  size_t bytes() const { return W.weight_bytes() + scale.size() * 2; }
};

struct Layer {
  TernLinear wq, wk, wv, wo;
  TernLinear w_gate, w_up, w_down;
  std::vector<uint16_t> rms_att, rms_ffn;   // [d_model]
  size_t bytes() const {
    return wq.bytes()+wk.bytes()+wv.bytes()+wo.bytes()
         + w_gate.bytes()+w_up.bytes()+w_down.bytes()
         + (rms_att.size()+rms_ffn.size())*2;
  }
};

// ---------------------------------------------------------------------------
//  Raw fp32 master weights (training side), mirroring the quantised model.
// ---------------------------------------------------------------------------
struct MasterWeights {
  std::vector<float> token_emb;     // [vocab][d]
  std::vector<float> rms_final;     // [d]
  struct L {
    std::vector<float> wq,wk,wv,wo;      // [out][in]
    std::vector<float> wg,wu,wd;
    std::vector<float> rms_att, rms_ffn;
  };
  std::vector<L> layers;
  std::vector<float> lm_head;       // [vocab][d]
  size_t numel() const;
};

// ---------------------------------------------------------------------------
//  Model: device residency + forward.
// ---------------------------------------------------------------------------
// test helpers
void fill_rand_f16(std::vector<uint16_t>& v, std::mt19937& rng, float sigma);
void ternarize_for_test(const std::vector<float>& w, int N, int K, TernaryMatrix& out);

class Model {
public:
  Config cfg;
  std::vector<Layer> layers;
  std::vector<uint16_t> token_emb;   // [vocab][d], fp16
  std::vector<uint16_t> rms_final;   // [d]
  std::vector<uint16_t> lm_head;     // [vocab][d], fp16 (tied to embeddings if tied)

  // ---- device buffers (managed by Device image) ----
  struct Impl;
  std::unique_ptr<Impl> impl;

  Model();
  ~Model();

  void init(const Config& c);
  void quantise_from(const MasterWeights& mw);   // absolute-mean ternary
  size_t ternary_weight_bytes() const;
  size_t total_bytes() const;
  bool   fits_in_l2(size_t l2_budget) const;
  void   validate() const;

  void to_device();
  // Reference forward using the TRAINER's kernels and fp32 precision.
  // Used by the alignment probe to prove whether the fp16 inference path
  // computes the same function the optimiser was differentiated against.
  void forward_fw(const std::vector<int>& ids, std::vector<float>& out_logits);

  // ---- inference ----
  void reset_kv();
  // Run one token through and return logits (host side, vocab floats).
  const std::vector<float>& forward(int token, int pos, bool use_cache);
  std::vector<float> last_logits_copy;    // snapshot of the most recent forward
  // Sample the next token from the last logits.
  int  sample(float temp, int top_k, float top_p, uint64_t seed);
  static int sample_from(const std::vector<float>& lg, float temp, int top_k,
                         float top_p, std::mt19937_64& rng);

  // ---- diagnostics ----
  std::string layer_type_name() const { return "ternary-2bit"; }
};

// ---------------------------------------------------------------------------
//  Byte-level BPE tokenizer, pure C++ (no Python, no external vocab).
// ---------------------------------------------------------------------------
class BPETokenizer {
public:
  std::vector<std::string> vocab;              // id -> byte string
  std::unordered_map<std::string, int> enc;    // byte string -> id
  std::vector<std::pair<int,int>> merges;      // rank order
  std::unordered_map<uint64_t,int> merge_rank;

  static uint64_t key(int a, int b) { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; }

  void train(const std::string& corpus, int target_vocab);
  std::vector<int> encode(const std::string& text) const;
  std::string decode(const std::vector<int>& ids) const;
  bool save(const std::string& path) const;
  bool load(const std::string& path);
  int  id_of(const std::string& s) const;
  void add_special(const std::string& s);
};

// ---------------------------------------------------------------------------
//  Chat + tool-use protocol.
// ---------------------------------------------------------------------------
struct ToolCall { std::string name; std::string args_json; };

// ---------------------------------------------------------------------------
//  Structured tool outcome.
//
//  WHY THIS REPLACES A BARE STRING
//  -------------------------------
//  Every tool used to return free text and the caller inferred success by
//  prefix-matching "error:" or parsing "exit=N" out of the body. That is fragile
//  in three separate ways: a tool whose legitimate output contains the word
//  "error" reads as a failure, a shell command returning non-zero is only
//  detected if the exit code happens to be printed first, and no caller can tell
//  "the operation worked" from "the operation ran and printed something".
//
//  Automatic correction needs a trustworthy failure signal -- that is the whole
//  premise of learning from mistakes -- so the outcome is now explicit:
//
//      ok        did the operation succeed
//      output    what the user should see (ORIGINAL-NODE: unchanged text)
//      error     short machine-readable reason when ok is false
//      fixable   true when retrying with a different payload could plausibly
//                succeed (a missing file), false when the request itself was
//                malformed (unparseable JSON). Only fixable failures are worth
//                learning from; the rest indicate a caller bug.
// ---------------------------------------------------------------------------
struct ToolResult {
  std::string name;
  std::string output;      // the original free text, unchanged
  bool ok = true;
  bool fixable = false;    // plausible to succeed with different arguments
  std::string error;       // machine-readable reason, empty when ok
};

// Format a conversation into the model's expected prompt template.
std::string render_chat(const std::vector<std::pair<std::string,std::string>>& turns,
                        const std::string& system_prompt);

// Render the conversation DIRECTLY into token ids, mirroring pack_samples().
//
// This exists because encoding "<|user|>" together with the following prompt
// lets BPE merge across the boundary and emit token sequences the model never
// saw in training. Training pushes the special token as its OWN id and encodes
// the payload separately; inference must do exactly the same or the model is
// fed an off-distribution byte pattern. That mismatch was the reason every
// prompt produced the same repeated garbage.
std::vector<int> render_chat_ids(const std::vector<std::pair<std::string,std::string>>& turns,
                                 const std::string& system_prompt,
                                 const BPETokenizer& tk);

// Parse `<tool>name {"arg":...}</tool>` blocks out of a model reply.
std::vector<ToolCall> parse_tool_calls(const std::string& text);
std::string strip_tool_calls(const std::string& text);

// Built in tools. Each returns a plain-text observation.
std::string tool_read_file(const std::string& args_json);
std::string tool_write_file(const std::string& args_json);
std::string tool_list_dir (const std::string& args_json);
std::string tool_run_shell(const std::string& args_json);
std::string tool_calc     (const std::string& args_json);
// Read-only additions. Both exist because these requests were previously in the
// Unsupported list -- correctly, since no tool answered them -- and the alternative
// (routing them to read_file) opens the file instead of describing it.
std::string tool_stat     (const std::string& args_json);
std::string tool_search   (const std::string& args_json);
// Additive only: creating a directory cannot destroy existing data.
std::string tool_mkdir    (const std::string& args_json);
// Also additive: the source is only read, and an existing destination is refused
// rather than replaced.
std::string tool_copy     (const std::string& args_json);

// ---------------------------------------------------------------------------
//  One entry per tool, and the ONLY place a tool is declared.
//
//  WHY THE FUNCTION POINTER IS PART OF THE SPEC
//  --------------------------------------------
//  The tool name previously appeared in FOUR hand-maintained lists:
//
//      tool_specs()          name + description + example     (the prompt)
//      dispatch_tool_result  a whitelist of accepted names
//      dispatch_tool         a name -> implementation chain
//      rewrite_multi_arg     the inheritance rebuild table      (am.cpp)
//
//  Every one of them had to be edited when a tool was added, and twice they were
//  not: defect 70 left search/stat/mkdir/copy out of the inheritance table (so
//  four tools silently lost multi-turn context) and the whitelist/dispatch pair
//  was flagged as a hazard in its own comment before it caused a bug.
//
//  "Remember to update four places" is not a rule that survives contact with a
//  growing system. Binding the implementation to the name in a single table makes
//  the other three derivable: dispatch is a lookup, the whitelist is membership,
//  and the prompt text is already here.
//
//  A tool is now added by ONE line in one array.
// ---------------------------------------------------------------------------
typedef std::string (*ToolFn)(const std::string& args_json);
struct ToolSpec {
  const char* name;
  const char* desc;
  const char* example;
  ToolFn fn;
  // The surface form used to REBUILD this tool's request when a follow-up turn
  // inherits it ("again", "and C:/other"). nullptr means the tool takes more than
  // one argument and cannot be rebuilt from a single inherited payload, in which
  // case inheritance re-runs the previous request verbatim.
  //
  // This field exists because its absence caused defect 70: the rebuild was a
  // separate if/else chain in am.cpp, four tools were missing from it, and the
  // consequence -- "again" after "find all TODO in C:/tmp" being REFUSED -- was
  // invisible until a generated hunt looked for it.
  //
  // Stated generally: when a fact about a tool is needed in two places, the fact
  // must live WITH the tool, not be restated where it is used.
  const char* lead;

  // The bucket this tool's episodes are stored under, as an int because model.h
  // cannot see the Intent enum (am.h defines it). Cast back at the use site.
  //
  // This was the THIRD hand-maintained name->something chain in the codebase
  // (defects 70 and 73 were the first two), in rebuild_buckets:
  //
  //     if      (t == "read_file")  in = Intent::ReadFile;
  //     else if (t == "list_dir")   in = Intent::ListDir;
  //     ...
  //
  // A tool missing from that chain is stored under Unknown, so retrieval silently
  // never looks in its bucket for structural queries -- the same class of invisible
  // failure as a missing rebuild lead.
  int intent_id;

  // A prompt that MUST infer to this tool. Used by am-source to prove that the
  // name, the implementation, the bucket and the intent detector all agree.
  //
  // This closes the loop between what the table claims and what infer_intent
  // actually decides. A wrong intent_id is not merely "an odd number" -- it makes
  // the bucket unreachable, and this catches that directly rather than trusting a
  // numeric convention.
  const char* canonical;
};
// The single table of tools. Every other view of "what tools exist" -- dispatch,
// the accepted-name check, the system prompt, and the multi-turn rebuild -- is
// derived from this one. Adding a tool is one line above and nothing else.
const std::vector<ToolSpec>& tool_specs();
std::string tool_system_prompt();
std::string dispatch_tool(const ToolCall& c);

// Same dispatch, with a structured outcome. Prefer this: the string version
// can only report failure by convention.
ToolResult dispatch_tool_result(const ToolCall& c);

// Minimal JSON helpers (no dependency on a JSON library).
std::string json_get_string(const std::string& obj, const std::string& key);
double      json_get_number(const std::string& obj, const std::string& key);

} // namespace canticle

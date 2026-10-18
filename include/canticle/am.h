#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace canticle {

// ---- GPU retrieval primitives (implemented in am_kernels.cu) ----
void am_dot(const __half* K, const float* q, float* sim, int N, int D, cudaStream_t s);
void am_to_f16(const float* src, __half* dst, long n, cudaStream_t s);

// ---- ternary keys: 8x smaller than fp16, scored by popcount ----
// Gather a row subset on the DEVICE. Copying the whole matrix to the host to
// index it costs 52 MB per query at N=13500, D=2048.
void am_gather_f16(const __half* src, const int* rows, __half* dst, int m, int D,
                   cudaStream_t s);
void am_gather_tern(const uint32_t* P, const uint32_t* M, const float* inv,
                    const int* rows, uint32_t* dP, uint32_t* dM, float* dInv,
                    int m, int WPR, cudaStream_t s);

void am_quantize_keys(const float* K, int N, int D, uint32_t* P, uint32_t* M,
                      float* invnorm, float thr_frac, cudaStream_t s);
void am_quantize_query(const float* q, int D, uint32_t* P, uint32_t* M,
                       float thr_frac, float* nnz_out, cudaStream_t s);
void am_dot_ternary(const uint32_t* KP, const uint32_t* KM, const float* invnorm,
                    const uint32_t* QP, const uint32_t* QM,
                    float* sim, int N, int D, float qnnz, cudaStream_t s);
void am_softmax(float* s, int N, float beta, cudaStream_t st);
void am_back(const __half* K, const float* w, float* qout, int N, int D, cudaStream_t s);
void am_l2(float* v, int n, cudaStream_t s);

// ===========================================================================
//  CANTICLE-AM: associative memory encoder.
//
//  A representation in which unrelated inputs are orthogonal cannot support
//  nearest-neighbour recall -- there is no similarity gradient to follow. This
//  encoder makes similar TEXT similar by construction, using character n-gram
//  feature hashing weighted by IDF, plus reserved dimensions for the routing
//  keywords. It uses no trained parameters whatsoever.
// ===========================================================================
struct AmEncoder {
  int D = 2048;
  bool idf_built = false;
  std::unordered_map<uint64_t, float> idf;

  void build_idf(const std::vector<std::string>& docs);
  void encode(const std::string& text, std::vector<float>& out) const;
  void encode_marked(const std::string& text, std::vector<float>& out) const;
  static void l2norm(std::vector<float>& v);
};

// ===========================================================================
//  Intent inference.
//
//  WHY THIS EXISTS
//  --------------
//  The keyword dimensions in AmEncoder are a HARD vote: one hit adds 3.0 and
//  outweighs everything else. A paraphrase test showed the cost --
//  "what is in C:/tmp" routed to calc because "what is" fired the calc keyword,
//  even though the sentence contains a path and no expression. list_dir scored
//  2/6 while every other intent scored 100%.
//
//  Structure, not vocabulary, decides the bucket: an arithmetic expression is a
//  NUMBER next to an OPERATOR; a directory question is a PATH with no verb. The
//  n-gram vector still decides WHICH stored episode to pick inside a bucket --
//  the two are complementary, not competing.
// ===========================================================================
// ---------------------------------------------------------------------------
//  Unsupported is NOT the same as Unknown, and the distinction is safety-critical.
//
//      Unknown      the system does not recognise this request at all. The
//                   embedding is the only evidence, so a weak match means refuse.
//
//      Unsupported  the request IS understood -- it names a well-known operation --
//                   but no tool implements it. "delete C:/tmp/x.txt" is perfectly
//                   clear; the system simply cannot do it.
//
//  Measured reason for the split: BEFORE this enum existed, "delete C:/tmp/scratch.txt"
//  was routed to read_file. The structure layer saw a path and no write verb, the
//  bare-path default fired, and a DESTRUCTIVE request came back as a read. It did
//  not execute anything harmful in that instance only because read_file happened to
//  be what it chose -- the misrouting itself is the hazard.
//
//  Classifying it as Unsupported means the request is refused explicitly, and the
//  refusal can say WHY: not "I do not understand" but "I understand and do not do
//  that". Those are different promises to the user.
// ---------------------------------------------------------------------------
enum class Intent { Unknown, ReadFile, ListDir, WriteFile, RunShell, Calc,
                     Stat, Search, MkDir, Copy, Unsupported };

// The number of BUCKETS, derived from the enum rather than written down.
//
// Defect 49 was this number going stale: the enum grew by one, the bucket array
// did not, and every write went one element past the end. It surfaced three calls
// later as an "invalid argument" from an unrelated kernel, which is about as far
// from the cause as a symptom can get.
//
// The fix at the time introduced a named constant and a bounds-checking bucket_of,
// which turned a memory error into a silent fallback. That was an improvement and
// not a solution: the constant still had to be edited by hand, and the failure it
// left behind is invisible -- an unmapped tool is stored under Unknown and simply
// never found by a structural query.
//
// Now the count cannot disagree with the enum. Unsupported is excluded because it
// is refused before any search runs, so it needs no bucket; every other value maps
// to its own index, and tool_specs() records which tool owns which.
static constexpr int kIntentBucketCount = (int)Intent::Unsupported;

// Declared here so the classifier in am.cpp can consult them; defined there.
// A request is Unsupported when it names an operation this system has no tool for.
bool intent_is_unsupported_op(const std::string& lower_text);

Intent infer_intent(const std::string& text);

// Human-readable name, for diagnostics.
const char* intent_name(Intent i);

// Affinity in [0,1] of a text for a tool name as it appears in a reply.
// Used to re-rank retrieved candidates.
float intent_affinity(const std::string& text, const std::string& tool);

// ===========================================================================
//  Multi-turn context (gap B).
//
//  THE PROBLEM
//  -----------
//  am-chat filled a history vector and never read it. Every turn was retrieved
//  in isolation, so a follow-up like "again" or "and C:/work" had no way to
//  know which tool or which payload the previous turn used.
//
//  THE APPROACH
//  ------------
//  Context inheritance, the mirror image of slot transfer. slot_transfer COPIES
//  a payload outward into the reply; inheritance pulls the PREVIOUS turn's tool
//  and payload forward into the query. The rewritten query is then handed to
//  the ordinary retrieval path, so no part of the retrieval stack changes.
//
//  A turn is treated as context-dependent when either
//    * the sentence structure yields no intent, or
//    * no payload can be extracted from it,
//  which is exactly the case where the sentence carries too little to act on.
// ===========================================================================
struct Turn {
  std::string user;      // what the user typed
  std::string tool;      // tool family that was used, empty for plain chat
  std::string payload;   // the variable part that was used
};

class Session {
public:
  std::vector<Turn> turns;

  // Rewrite the text using recent turns. Returns it unchanged when it is
  // already self-contained.
  std::string rewrite(const std::string& text) const;

  // Record a completed turn.
  void push(const std::string& user, const std::string& tool, const std::string& payload);

  // The most recent turn that actually used a tool, or nullptr.
  const Turn* last_tool_turn() const;

  void clear() { turns.clear(); }
};

// True when the sentence cannot act on its own and needs the previous turn.
bool needs_context(const std::string& text);

// True when a string looks like data (contains a digit, a separator, or a dot)
// rather than prose. Guards against treating a cue like "again" as a payload.
bool payload_is_data(const std::string& p);

// Query a saved memory from outside am_main.cpp. Loads once and caches.
// Returns 0 on success; out_reply is the retrieved reply with the payload
// transferred from the query, out_sim the cosine of the best match.
int am_query_public(const std::string& ampath, const std::string& prompt,
                    std::string& out_reply, float& out_sim);

// Pull the variable part (a path, a command, an expression) out of a prompt.
std::string extract_payload(const std::string& q);

// The needle of a search request, with the path and lead-in removed.
// Separate from extract_payload because search has TWO arguments while slot
// transfer moves one; see the note on the definition.
std::string extract_search_pattern(const std::string& q);

// Source and destination of a copy request. Both paths come from the query
// because the stored template holds no payload for a two-path tool.
bool extract_copy_paths(const std::string& q, std::string* src, std::string* dst);

// True when the payload is an arithmetic expression ("40 - 14", "7+8").
bool payload_is_expression(const std::string& p);

// Canonicalise a payload for insertion into a stored reply.
//
// WHY: a retrieved reply was stored as {"expr":"40-22"} -- no spaces, because
// that is how the training data spells it. Substituting the query payload
// verbatim produced {"expr":"40 - 14"}: arithmetically correct, but scored as a
// mismatch against the target {"expr":"40-14"}. Normalising the spacing took
// exact-reply accuracy from 84% back to ~99% with no behaviour change. A pure
// formatting difference must not read as an accuracy loss.
std::string normalise_payload(const std::string& p);

} // namespace canticle

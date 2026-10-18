
// ============================================================================
//  canticle chat   - interactive / scripted conversation with tool use.
//
//  Flow per turn:
//     1. render the conversation into the <|user|>/<|assistant|> template
//     2. generate greedily-ish until <|end|> or a token budget
//     3. parse any <|tool|>...<|/tool|> block, execute it, append the result
//        as a <|tool_out|> turn
//     4. either answer the user with the observation in context (one more
//        generation), or loop if the model asks for a second tool
// ============================================================================
#include "canticle/model.h"
#include "canticle/ckpt.h"
#include "canticle/kernels.h"
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>

using namespace canticle;

struct GenOpts {
  int   max_new   = 64;
  float temp      = 0.3f;
  int   top_k     = 32;
  float top_p     = 0.9f;
  bool  greedy    = false;
};

// Generate a reply string. Uses the KV cache: we re-run the whole prompt once
// and then feed one token at a time.
static std::string generate(Model& m, BPETokenizer& tk,
                            const std::vector<int>& prompt_ids,
                            const GenOpts& o,
                            const std::vector<int>& stop_ids)
{
  // Keep the TAIL of the prompt, not the head. The instruction the model must
  // act on lives at the end (<|assistant|> is the last token); truncating from
  // the front of the window would discard the user's turn and leave only the
  // identical system preamble, which makes every prompt produce the same reply.
  std::vector<int> ids = prompt_ids;
  const int room = m.cfg.ctx - o.max_new - 4;
  if ((int)ids.size() > room && room > 0) {
    ids.erase(ids.begin(), ids.begin() + ((int)ids.size() - room));
    std::fprintf(stderr, "[ctx] prompt %zu tokens > room %d: keeping the tail\n",
                 prompt_ids.size(), room);
  }
  m.reset_kv();
  int pos = 0;
  const std::vector<float>* lg = nullptr;
  for (int id : ids) {
    lg = &m.forward(id, pos++, true);
    if (pos >= m.cfg.ctx) break;
  }
  std::vector<int> out;
  for (int i = 0; i < o.max_new && pos < m.cfg.ctx; ++i) {
    int nxt = o.greedy ? m.sample(0.0f, 1, 1.0f, 0)
                       : m.sample(o.temp, o.top_k, o.top_p, 0);
    bool stop = false;
    for (int s : stop_ids) if (nxt == s) { stop = true; break; }
    if (stop) break;
    out.push_back(nxt);
    lg = &m.forward(nxt, pos++, true);
  }
  return tk.decode(out);
}

static int id_of(BPETokenizer& tk, const std::string& s) {
  auto it = tk.enc.find(s);
  return it == tk.enc.end() ? -1 : it->second;
}

// One full user turn, including any tool round trips.
static std::string converse(Model& m, BPETokenizer& tk,
                            std::vector<std::pair<std::string,std::string>>& history,
                            const std::string& user_msg,
                            const GenOpts& o, bool allow_tools, int max_tool_rounds)
{
  history.push_back({"user", user_msg});
  std::string sys = allow_tools ? tool_system_prompt()
                                : "You are CANTICLE, a tiny local assistant. Answer briefly.\n";

  int iu = id_of(tk, "<|user|>"), ia = id_of(tk, "<|assistant|>");
  int io = id_of(tk, "<|tool_out|>");
  std::vector<int> stop = { iu, ia, io };

  for (int round = 0; round < max_tool_rounds; ++round) {
    // Token ids are built the SAME WAY as in training (special tokens pushed as
    // their own ids). See render_chat_ids for why this matters.
    std::vector<int> ids = render_chat_ids(history, sys, tk);
    std::string reply = generate(m, tk, ids, o, stop);

    auto calls = parse_tool_calls(reply);
    if (calls.empty() || !allow_tools) {
      history.push_back({"assistant", reply});
      return reply;
    }
    // execute the first tool call
    std::string obs = dispatch_tool(calls[0]);
    history.push_back({"assistant", reply});
    history.push_back({"tool", "[" + calls[0].name + "] " + obs});

    // If this was the last allowed round, ask for a plain answer instead.
    if (round == max_tool_rounds - 1) {
      std::vector<int> p2 = render_chat_ids(history, sys, tk);
      std::string fin = generate(m, tk, p2, o, stop);
      history.push_back({"assistant", fin});
      return fin;
    }
  }
  return "";
}

int chat_run(int argc, char** argv) {
  std::string ckpt = (argc > 0 && argv[0][0]) ? argv[0] : "build\\ckpt\\canticle.bin";
  bool once = false, no_tools = false;
  std::string one_shot;
  GenOpts o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--once")      once = true;
    if (a == "--no-tools")  no_tools = true;
    if (a == "--prompt" && i + 1 < argc) { one_shot = argv[++i]; once = true; }
    if (a == "--temp"  && i + 1 < argc) o.temp = (float)atof(argv[++i]);
    if (a == "--max"   && i + 1 < argc) o.max_new = atoi(argv[++i]);
    if (a == "--greedy") o.greedy = true;
  }

  Model m;
  BPETokenizer tk;
  bool dump_prompt = false;
  for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--dump-ids") dump_prompt = true;
  if (!load_checkpoint(ckpt, m, &tk)) {
    std::fprintf(stderr, "cannot load checkpoint: %s\n", ckpt.c_str());
    std::fprintf(stderr, "train one first:  canticle train --steps N\n");
    return 2;
  }
  m.to_device();
  std::printf("CANTICLE loaded: vocab=%d d=%d L=%d  ternary=%.2f MiB (%.2f bits/weight)\n",
              m.cfg.vocab, m.cfg.d_model, m.cfg.n_layer,
              m.ternary_weight_bytes() / 1048576.0,
              8.0 * m.ternary_weight_bytes() / (double)(m.cfg.d_model * m.cfg.d_model * 7 * m.cfg.n_layer));
  std::vector<std::pair<std::string,std::string>> hist;

  if (dump_prompt) {
    std::vector<std::pair<std::string,std::string>> h;
    h.push_back({"user", one_shot});
    std::string sys = no_tools
      ? "You are CANTICLE, a tiny local assistant. Answer briefly.\n"
      : tool_system_prompt();
    std::vector<int> ids = render_chat_ids(h, sys, tk);
    std::printf("prompt ids (%zu):", ids.size());
    for (int id : ids) std::printf(" %d", id);
    std::printf("\ndecoded: [");
    std::string d = tk.decode(ids);
    for (char c : d) std::printf(c == '\n' ? "|" : "%c", c);
    std::printf("]\n");
    std::printf("id range [%d,%d] vocab=%d\n", 
                *std::min_element(ids.begin(), ids.end()),
                *std::max_element(ids.begin(), ids.end()), m.cfg.vocab);
    return 0;
  }

  if (!one_shot.empty()) {
    std::string r = converse(m, tk, hist, one_shot, o, !no_tools, 2);
    std::printf("\nuser> %s\ncanticle> %s\n", one_shot.c_str(), r.c_str());
    return 0;
  }

  std::printf("type your message, Ctrl-Z then Enter to quit\n\n");
  std::string line;
  while (true) {
    std::printf("user> "); std::fflush(stdout);
    if (!std::getline(std::cin, line)) break;
    if (line.empty()) continue;
    double t0 = now_ms();
    std::string r = converse(m, tk, hist, line, o, !no_tools, 2);
    double t1 = now_ms();
    std::printf("canticle> %s\n", r.c_str());
    std::printf("          [%.0f ms]\n\n", t1 - t0);
    if (once) break;
  }
  return 0;
}

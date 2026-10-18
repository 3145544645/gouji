
// Decode a few packed training sequences so the structure can be eyeballed.
#include "canticle/ckpt.h"
#include "canticle/ftrain.h"
#include "canticle/data.h"
#include <cstdio>
using namespace canticle;

int dump_run() {
  BPETokenizer tk;
  Model m;
  if (!load_checkpoint("build\\ckpt\\canticle.bin", m, &tk)) {
    std::printf("no ckpt\n"); return 1;
  }
  std::printf("vocab=%zu merges=%zu\n", tk.vocab.size(), tk.merges.size());
  for (const char* s : {"<pad>", "<unk>", "<|user|>", "<|assistant|>", "<|tool|>", "<|/tool|>"}) {
    auto it = tk.enc.find(s);
    std::printf("  special %-14s -> %d\n", s, it == tk.enc.end() ? -999 : it->second);
  }

  auto samples = synth_tool_data(12, 2026);
  auto seqs = pack_samples(samples, tk, tool_system_prompt(), 128);
  std::printf("packed %zu seqs\n\n", seqs.size());
  for (int i = 0; i < 3 && i < (int)seqs.size(); ++i) {
    const auto& sq = seqs[i];
    std::vector<int> body(sq.ids.begin(), sq.ids.begin() + 100);
    std::string txt = tk.decode(body);
    std::printf("--- seq %d (first 100 of %zu tokens) ---\n", i, sq.ids.size());
    // show with markers visible
    for (char& c : txt) if (c == '\n') c = '|';
    std::printf("%s\n\n", txt.c_str());
    std::printf("targets[40..70]:");
    for (int j = 40; j < 70 && j + 1 < (int)sq.ids.size(); ++j) std::printf(" %d", sq.ids[j+1]);
    std::printf("\n\n");
  }
  return 0;
}

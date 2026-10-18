
#include "canticle/ckpt.h"
#include "canticle/data.h"
#include <cstdio>
using namespace canticle;

int tokdiag_run() {
  auto samples = synth_tool_data(200, 1);
  std::string corpus;
  for (auto& s : samples) corpus += "<|user|>\n" + s.prompt + "\n<|assistant|>\n" + s.reply + "\n";
  corpus += tool_system_prompt();

  BPETokenizer a;
  a.train(corpus, 2048);
  std::printf("after train: vocab=%zu merges=%zu\n", a.vocab.size(), a.merges.size());
  for (const char* s : {"<pad>", "<unk>", "<|user|>", "<|assistant|>", "<|tool|>", "<|/tool|>"}) {
    auto it = a.enc.find(s);
    std::printf("  %-14s -> %d\n", s, it == a.enc.end() ? -999 : it->second);
  }
  std::printf("  vocab[0..7]: ");
  for (int i = 0; i < 8 && i < (int)a.vocab.size(); ++i) std::printf("[%d]%s ", i, a.vocab[i].c_str());
  std::printf("\n");

  // round trip through save/load
  Model m; Config c; c.vocab = (int)a.vocab.size(); c.d_model=64; c.n_layer=1;
  c.n_head=2; c.d_ff=128; c.ctx=64;
  m.init(c);
  // push the tokenizer's own content into the model's fp16 tables is irrelevant;
  // we only care that save_checkpoint writes a.loadable tokenizer
  if (!save_checkpoint("build\\ckpt\\toktest.bin", m, &a)) { std::printf("save failed\n"); return 1; }

  BPETokenizer b; Model m2;
  if (!load_checkpoint("build\\ckpt\\toktest.bin", m2, &b)) { std::printf("load failed\n"); return 1; }
  std::printf("\nafter load: vocab=%zu merges=%zu\n", b.vocab.size(), b.merges.size());
  for (const char* s : {"<pad>", "<unk>", "<|user|>", "<|assistant|>", "<|tool|>", "<|/tool|>"}) {
    auto it = b.enc.find(s);
    std::printf("  %-14s -> %d\n", s, it == b.enc.end() ? -999 : it->second);
  }
  std::printf("  vocab[0..7]: ");
  for (int i = 0; i < 8 && i < (int)b.vocab.size(); ++i) std::printf("[%d]%s ", i, b.vocab[i].c_str());
  std::printf("\n");

  // encode equivalence
  std::string probe = "read the file C:/work/gamma.txt";
  auto ea = a.encode(probe), eb = b.encode(probe);
  std::printf("\nencode equal: %s (a=%zu ids, b=%zu ids)\n",
              (ea == eb) ? "YES" : "NO", ea.size(), eb.size());
  return 0;
}

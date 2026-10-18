
// ===========================================================================
//  Train/inference forward alignment probe.
//
//  THE QUESTION: model.cpp's inference forward and ftrain_main.cpp's training
//  forward are DIFFERENT CODE PATHS with different kernels, different
//  precision (fp16 activations vs fp32) and two separate attention
//  implementations. If they disagree, the model optimises one function and is
//  evaluated on another -- which would look exactly like "training loss falls
//  but the model talks nonsense".
//
//  This probe runs the same token sequence through both and compares logits.
// ===========================================================================
#include "canticle/ckpt.h"
#include "canticle/ftrain.h"
#include "canticle/ops.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace canticle;

// The training forward lives in ftrain_main.cpp inside namespace canticle.
namespace canticle {
int train_forward_logits(Model& m, const std::vector<int>& ids,
                         std::vector<float>& out_logits);
}

int align_run() {
  const char* path = "build\\ckpt\\canticle.bin";
  Model m; BPETokenizer tk;
  if (!load_checkpoint(path, m, &tk)) { std::printf("no ckpt\n"); return 1; }
  m.to_device();

  // a short sequence so both paths see identical input
  std::vector<int> ids = tk.encode("<|system|>\n" + tool_system_prompt() + "\n");
  auto u = tk.enc.find("<|user|>");
  if (u != tk.enc.end()) ids.push_back(u->second);
  for (int id : tk.encode("read the file C:/work/gamma.txt")) ids.push_back(id);
  auto a = tk.enc.find("<|assistant|>");
  if (a != tk.enc.end()) ids.push_back(a->second);

  const int L = (int)ids.size();
  std::printf("=== train/inference forward alignment ===\n");
  std::printf("sequence: %d tokens, vocab=%d\n\n", L, m.cfg.vocab);

  // ---- path 1: inference forward (KV cache, fp16 activations) ----
  std::vector<float> lg_inf;
  {
    m.reset_kv();
    for (int i = 0; i < L; ++i) {
      const auto& lg = m.forward(ids[i], i, true);
      if (i == L - 1) lg_inf = lg;
    }
  }

  // ---- path 2: training forward (full sequence, fp32 activations) ----
  std::vector<float> lg_tr;
  if (canticle::train_forward_logits(m, ids, lg_tr) != 0) {
    std::printf("training forward hook failed\n");
    return 1;
  }

  std::printf("inference logits: n=%zu\n", lg_inf.size());
  std::printf("training  logits: n=%zu\n", lg_tr.size());
  if (lg_inf.size() != lg_tr.size()) { std::printf("SIZE MISMATCH\n"); return 1; }

  // ---- compare ----
  double mx = 0, sum = 0; int nan_i = 0, nan_t = 0;
  for (size_t i = 0; i < lg_inf.size(); ++i) {
    if (lg_inf[i] != lg_inf[i]) nan_i++;
    if (lg_tr[i] != lg_tr[i]) nan_t++;
    if (lg_inf[i] == lg_inf[i] && lg_tr[i] == lg_tr[i])
      mx = std::max(mx, (double)std::fabs(lg_inf[i] - lg_tr[i]));
    sum += std::fabs(lg_tr[i]);
  }
  const double mean_abs = sum / lg_tr.size();

  auto argmax = [](const std::vector<float>& v) {
    int b = 0; float bv = -1e30f;
    for (size_t i = 0; i < v.size(); ++i) if (v[i] > bv) { bv = v[i]; b = (int)i; }
    return b;
  };
  int ai = argmax(lg_inf), at = argmax(lg_tr);

  std::printf("\nNaN: inference=%d training=%d\n", nan_i, nan_t);
  std::printf("max |diff|      : %.6f\n", mx);
  std::printf("mean |logit|    : %.6f\n", mean_abs);
  std::printf("relative diff   : %.4f  (%.2f%%)\n", mx / (mean_abs + 1e-9),
              100.0 * mx / (mean_abs + 1e-9));
  std::printf("argmax infer    : %d\n", ai);
  std::printf("argmax train    : %d\n", at);
  std::printf("argmax AGREE    : %s\n", ai == at ? "YES" : "NO");

  // top-5 overlap
  auto top5 = [](const std::vector<float>& v) {
    std::vector<int> idx(v.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int x, int y) { return v[x] > v[y]; });
    idx.resize(5);
    return idx;
  };
  auto ti = top5(lg_inf), tt = top5(lg_tr);
  int ov = 0;
  for (int x : ti) for (int y : tt) if (x == y) ov++;
  std::printf("top-5 overlap   : %d/5\n", ov);

  std::printf("\n--- sample of the two distributions ---\n");
  std::printf("%8s %14s %14s\n", "token", "inference", "training");
  for (int k = 0; k < 8; ++k) {
    std::printf("%8d %14.5f %14.5f\n", ti[k], lg_inf[ti[k]], lg_tr[ti[k]]);
  }

  std::printf("\n[verdict] %s\n",
              (mx / (mean_abs + 1e-9) < 0.05 && ai == at)
                ? "PATHS AGREE -- training/inference mismatch is NOT the cause"
                : "PATHS DISAGREE -- this is a real training/inference gap");

  // what does the training path actually predict?
  std::printf("\ntraining-path prediction decodes to: [%s]\n",
              tk.decode({ at }).c_str());
  std::printf("inference-path prediction decodes to: [%s]\n",
              tk.decode({ ai }).c_str());
  return 0;
}

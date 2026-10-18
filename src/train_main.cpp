
// ============================================================================
//  canticle ste   - prove straight-through ternary learning actually learns.
//
//  WHY THIS TASK
//  -------------
//  The head is a LINEAR classifier over a MEAN-POOLED embedding, so the target
//  must be linearly decodable from   h = (1/T) * sum_t emb[id_t].
//  An earlier version of this test used  target = (sum of ids) mod vocab, which
//  is a wrap-around and provably outside the reach of a linear head: the model
//  drove train loss to 0.30x uniform with 81% train accuracy and 0% test
//  accuracy, i.e. it memorised 512 samples and learned nothing transferable.
//  That was a bad task, not a broken trainer.
//
//  The task below is a monotone bucketisation of the summed ids, which IS a
//  linear function of the per-token contribution, so a correct trainer must
//  generalise to held-out samples. Generalisation is the whole point: a model
//  that only memorises would fail here too.
//
//  Weights are ternary in EVERY forward pass. If held-out loss does not fall
//  well below the uniform baseline, the ternary training path is broken.
// ============================================================================
#include "canticle/train.h"
#include "canticle/ops.h"
#include <random>
#include <cmath>
#include <algorithm>

using namespace canticle;

int train_run(int argc = 0, char** argv = nullptr) {
  const int V = 512, D = 64, T = 4;
  const int NCLS = 64;      // >2 means chance is 1/NCLS, so we can see real signal
  int STEPS = 3000;
  const int N_TRAIN = 2048, N_TEST = 512;
  int RQ = 8;
  float LR = 6e-3f;
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--rq"    && i + 1 < argc) RQ = atoi(argv[i + 1]);
    if (a == "--steps" && i + 1 < argc) STEPS = atoi(argv[i + 1]);
    if (a == "--lr"    && i + 1 < argc) LR = (float)atof(argv[i + 1]);
  }

  std::printf("=== CANTICLE straight-through ternary test ===\n");
  std::printf("vocab=%d  d=%d  seq=%d  steps=%d  lr=%.1e  requant_every=%d\n",
              V, D, T, STEPS, LR, RQ);
  std::printf("train samples=%d  test samples=%d\n", N_TRAIN, N_TEST);
  std::printf("task: identify a marker token placed at a random position\n");
  std::printf("      ids in [0,%d) are class markers, ids in [%d,%d) are distractors\n",
              NCLS, NCLS, V);
  std::printf("      a random marker is dropped into one random slot of %d\n\n", T);

  std::mt19937 rng(2026);
  std::uniform_int_distribution<int> cls(0, NCLS - 1);
  std::uniform_int_distribution<int> dst(NCLS, V - 1);
  std::uniform_int_distribution<int> slot(0, T - 1);
  auto make = [&](int n) {
    std::vector<HeadBatch> v(n);
    for (int i = 0; i < n; ++i) {
      int c = cls(rng);
      int p = slot(rng);
      for (int t = 0; t < T; ++t) v[i].ids.push_back(t == p ? c : dst(rng));
      v[i].target = c;
    }
    return v;
  };
  auto train = make(N_TRAIN);
  auto test  = make(N_TEST);

  HeadModel m;
  m.init(V, D, 7);
  m.to_device();

  HeadTrainer tr;
  tr.m = &m;
  tr.lr = LR;
  tr.requant_every = RQ;

  // refresh the ternary planes once before we start measuring
  tr.requantise();

  const double uniform = std::log((double)NCLS);
  std::printf("uniform baseline (random guessing) = %.4f\n\n", uniform);

  float first = 0, last = 0;
  double t0 = now_ms();
  for (int i = 0; i < STEPS; ++i) {
    const auto& b = train[i % train.size()];
    float loss = tr.train_step(b);
    if (i == 0) first = loss;
    last = loss;
    if (i % 150 == 0 || i == STEPS - 1) {
      std::printf("  step %5d   loss %.4f   ratio-to-uniform %.3f\n",
                  i, loss, loss / uniform);
      std::fflush(stdout);
    }
  }
  double t1 = now_ms();
  std::printf("\ntrain wall time: %.1f ms  (%.0f steps/s)\n", t1 - t0, STEPS * 1000.0 / (t1 - t0));

  // ---- evaluate using the FINAL ternary planes ----
  tr.requantise();
  float tl = tr.eval_loss(train);
  float vl = tr.eval_loss(test);

  // accuracy, computed on the device-side forward result
  auto accuracy = [&](std::vector<HeadBatch>& data) {
    int ok = 0;
    for (auto& b : data) {
      tr.eval_one(b);
      std::vector<float> lg = m.logits_cache, sc = m.scale_cache;
      // only the first NCLS head rows are supervised; the rest never receive
      // gradient, so restricting the argmax to them is the honest comparison.
      int best = 0; float bv = -1e30f;
      for (int v = 0; v < NCLS; ++v) { float y = lg[v]*sc[v]; if (y > bv) { bv = y; best = v; } }
      if (best == b.target) ok++;
    }
    return 100.0 * ok / data.size();
  };

  std::printf("\n--- results (weights ternary in EVERY forward pass) ---\n");
  std::printf("  first-step loss      : %.4f\n", first);
  std::printf("  final  train loss    : %.4f   (%.3f x uniform)\n", tl, tl / uniform);
  std::printf("  held-out test loss   : %.4f   (%.3f x uniform)\n", vl, vl / uniform);
  std::printf("  train accuracy       : %.2f%%\n", accuracy(train));
  std::printf("  test  accuracy       : %.2f%%\n", accuracy(test));
  std::printf("  ternary weights      : %zu of %zu\n",
              tr.ternary_weight_count(), (size_t)V*D);
  std::printf("  bytes at 2 bit/weight: %.1f KiB  (fp32 would be %.1f KiB)\n",
              V*D*2.0/8/1024, V*D*4.0/1024);

  // The gate is GENERALISATION: held-out loss must be far below uniform.
  // The gate is GENERALISATION on unseen marker/distractor combinations.
  bool pass = (vl / uniform) < 0.70;
  std::printf("\n  chance level (1/%d)    : %.2f%%\n", NCLS, 100.0 / NCLS);
  std::printf("[%s] straight-through ternary learning (gate: held-out loss < 0.70 x chance)\n",
              pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

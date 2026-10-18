#pragma once
#include "canticle/model.h"
#include "canticle/kernels.h"
#include <random>

namespace canticle {

// ===========================================================================
//  Training side.
//
//  Innovation #3: straight-through ternary learning.
//  The forward pass holds weights in {-1,0,+1} (2 bits). The backward pass
//  treats the quantiser as the identity, so gradients flow onto a full-precision
//  MASTER copy of the weights. The master is re-ternarised every K steps.
//  One ternary weight byte costs 2 bits, so 4x more weights fit in the same
//  256M-param fp32 budget -> more model per byte, trained without a GPU farm.
//
//  Head-only pre-training is the cheap sanity gate: if this cannot drive the
//  loss down, the whole training path is wrong and nothing downstream matters.
// ===========================================================================
struct HeadModel {
  int vocab = 0, d = 0;
  std::vector<float> emb;        // [vocab][d] master weights (fp32)
  std::vector<float> head;       // [vocab][d] master weights (fp32), ternary in forward
  std::vector<float> rms;        // [d]
  std::vector<float> adam_emb_m, adam_emb_v;
  std::vector<float> adam_head_m, adam_head_v;

  HeadModel();
  ~HeadModel();
  HeadModel(HeadModel&&) noexcept;
  HeadModel& operator=(HeadModel&&) noexcept;

  void init(int vocab_, int d_, uint64_t seed);

  struct Dev;
  std::unique_ptr<Dev> dev;
  void to_device();
  std::vector<float> logits_cache;
  std::vector<float> scale_cache;
};

// ---- device-side forward/backward for a mean-pooled head ----
struct HeadBatch {
  std::vector<int> ids;          // input token ids
  int target = 0;                // next-token target
};

class HeadTrainer {
public:
  HeadModel* m = nullptr;
  float lr = 3e-3f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
  int step = 0;
  int requant_every = 10;

  // one SGD step; returns the loss
  float train_step(const HeadBatch& b);
  float eval_one(const HeadBatch& b);
  // Forward only; writes scaled logits and the per-row scales into the given
  // vectors. Returns the loss. Used for accuracy accounting without exposing
  // device internals.
  float eval_one_to(std::vector<float>& logits_out, std::vector<float>& scale_out);
  // Re-derive the packed ternary planes from the fp32 master weights.
  void requantise();
  float eval_loss(const std::vector<HeadBatch>& data);
  // number of ternary weights kept in {-1,0,+1} in the forward pass
  size_t ternary_weight_count() const {
    return m ? (size_t)m->vocab * m->d : 0;
  }
};

} // namespace canticle

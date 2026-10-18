#pragma once
#include "canticle/model.h"
#include "canticle/data.h"
#include "canticle/kernels.h"
#include <random>

namespace canticle {

// ===========================================================================
//  Full-model trainer configuration.
//
//  Scope, deliberately narrow: the signal is next-token prediction on short
//  synthetic instruction/reply pairs. The goal is NOT a model that reasons --
//  it is a model that emits the tool-call protocol correctly, at a decode
//  speed only a 2-bit weight format makes possible on this hardware.
//
//  Trained:
//    token embeddings    fp32 master, AdamW
//    ternary linears     fp32 master + STE, re-ternarised every K steps
//    rmsnorm gains       fp32 master, AdamW
//  Not trained: nothing else exists -- every other op is parameter free.
// ===========================================================================
struct TrainCfg {
  int   steps        = 3000;
  int   seq_len      = 128;
  float lr           = 3e-3f;
  float wd           = 0.01f;
  int   requant_every = 20;
  int   log_every    = 25;
  int   seed         = 2026;
  int   n_samples    = 3000;
  int   vocab        = 2048;
  int   d_model      = 256;
  int   n_layer      = 4;
  int   n_head       = 4;
  int   d_ff         = 768;
  int   ctx          = 320;    // must be >= seq_len
  float clip         = 1.0f;    // element-wise |grad| clamp; 0 = off
};

// One packed training sequence: ids[0..L] with targets ids[1..L].
struct Seq { std::vector<int> ids; };

std::vector<Seq> pack_samples(const std::vector<Sample>& s, const BPETokenizer& tk,
                              const std::string& sys, int seq_len);

int full_train_run(int argc, char** argv);

} // namespace canticle

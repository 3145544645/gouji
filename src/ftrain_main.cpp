
// ============================================================================
//  canticle train  -  full-model straight-through ternary training.
//
//  Correctness notes that actually bit during development:
//    * ONE sequence at a time, gradients accumulated over the batch, then a
//      single AdamW update. Batching activations would need per-sequence
//      attention masking; this is simpler and mathematically identical.
//    * Several kernels use += or atomicAdd (rmsnorm and attention both have
//      fan-out), so every gradient buffer is explicitly zeroed first.
//    * The fp32 masters are what Adam updates. Packed (P,M) planes are
//      re-derived from them every few steps.
//    * dX from the ternary backward is ASSIGNED, not accumulated, because each
//      (row,k) element is owned by exactly one thread. dW is accumulated
//      because many (m) rows contribute to each weight.
// ============================================================================
#include "canticle/ftrain.h"
#include "canticle/ckpt.h"
#include "canticle/data.h"
#include "canticle/ops.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <filesystem>
#include <fstream>

namespace canticle {

// ------------------------------------------------------------------ packing
// ===========================================================================
//  Packing. ONE SAMPLE == ONE SEQUENCE, padded to a fixed length.
//
//  WHY THIS CHANGED
//  ----------------
//  The first version concatenated the whole corpus into one long stream and
//  then sliced it into fixed windows. Almost every window therefore began in
//  the middle of a sentence: the model never once observed a complete
//  "<|system|> ... <|user|> q <|assistant|> a" tuple, and at inference time it
//  degenerated into emitting one repeated token pattern regardless of the
//  prompt. The lesson generalises: for instruction tuning the UNIT of packing
//  has to be the instruction, not the byte stream.
//
//  Layout: system marker, newline, system text, newline, user marker, prompt,
//  assistant marker, reply, assistant marker (end of reply), then pad tokens.
//  Loss is masked on the pad positions by pointing their target at <|pad|>.
// ===========================================================================
static std::vector<Seq> pack_samples_impl(const std::vector<Sample>& samples, const BPETokenizer& tk,
                                          const std::string& sys, int seq_len, int limit) {
  std::vector<Seq> out;
  auto id_of = [&](const char* s) { auto it = tk.enc.find(s); return it == tk.enc.end() ? -1 : it->second; };
  const int iu = id_of("<|user|>"), ia = id_of("<|assistant|>");
  const int ip = id_of("<pad>");   // token 0, registered by BPETokenizer::train
  if (iu < 0 || ia < 0 || ip < 0) {
    std::fprintf(stderr, "[pack] special tokens missing from the tokenizer\n");
    return out;
  }

  std::vector<int> sysids = tk.encode("<|system|>\n" + sys + "\n");

  // LENGTH BUG THAT CAUSED A CUDA ILLEGAL MEMORY ACCESS
  // ---------------------------------------------------
  // Every sequence needs seq_len+1 ids: inputs at 0..seq_len-1 and the targets
  // at 1..seq_len. The first version built only seq_len of them, but the
  // training step reads sq.ids[i+1] for i in 0..L-1, i.e. index L. That is one
  // int PAST the end of the vector.
  //
  // The over-read itself is a host-side read of 4 bytes, which is why it went
  // unnoticed: beyond-the-end heap memory usually holds a small number, which
  // happens to be a valid token id, so training looked fine on small corpora.
  // Once the corpus grew, the heap layout changed and the garbage id exceeded
  // the vocabulary; k_ce_rows then indexed lg[tgt] outside the logit row and
  // the device faulted. Measured threshold: data<=4000 fine, data=6000 crashes.
  //
  // Fix: reserve one extra slot so the final target always exists.
  const int body = seq_len;                  // ids holds seq_len+1 entries

  for (const auto& s : samples) {
    if (limit > 0 && (int)out.size() >= limit) break;
    std::vector<int> ids;
    ids.reserve(body + 1);
    ids.insert(ids.end(), sysids.begin(), sysids.end());
    ids.push_back(iu);
    for (int id : tk.encode(s.prompt)) ids.push_back(id);
    ids.push_back(ia);
    for (int id : tk.encode(s.reply)) ids.push_back(id);
    ids.push_back(ia);
    if ((int)ids.size() > body) continue;   // drop over-long samples cleanly
    while ((int)ids.size() < body) ids.push_back(ip);
    ids.push_back(ia);                       // slot whose TARGET is the next pred
    if ((int)ids.size() != seq_len + 1) {    // the invariant the trainer relies on
      std::fprintf(stderr, "[pack] internal length error: %zu != %d\n",
                   ids.size(), seq_len + 1);
      return out;
    }
    Seq sq; sq.ids = std::move(ids);
    out.push_back(std::move(sq));
  }
  return out;
}

std::vector<Seq> pack_samples(const std::vector<Sample>& samples, const BPETokenizer& tk,
                              const std::string& sys, int seq_len) {
  return pack_samples_impl(samples, tk, sys, seq_len, 0);
}

static constexpr int NQT = 7;   // q k v o gate up down

struct LayerDev {
  GpuBuf W[NQT], dW[NQT], am[NQT], av[NQT];
  GpuBuf P[NQT], M[NQT], scl[NQT];
  GpuBuf rms_att, rms_ffn, d_rms_att, d_rms_ffn;
  GpuBuf am_ra, av_ra, am_rf, av_rf;
  GpuBuf xn_att, rrms_att, q, k, v, o, probs, x_mid;
  GpuBuf xn_ffn, rrms_ffn, gate, up, sig, hmid;
};

struct TrainerImpl {
  Model* m = nullptr;
  BPETokenizer* tk = nullptr;
  Config c;
  int d = 0, ff = 0, H = 0, hd = 0, V = 0, NL = 0, L = 0;
  std::vector<LayerDev> lay;

  GpuBuf emb_f32, emb_dW, emb_am, emb_av;
  GpuBuf head_f32, head_dW, head_am, head_av;
  GpuBuf rmsf, d_rmsf, rmsf_am, rmsf_av;

  GpuBuf x, dx, dxb, xn, dtmp, dtmp2, rrms, d_rrms;
  GpuBuf dq, dk, dv, datt, dprobs;
  GpuBuf dgate, dup, dhmid, dff, dffb, dxff;
  // bw_rmsnorm writes its dx with +=, so it needs a buffer that is zeroed
  // first. Passing the activation buffer T.dx (which holds the forward value)
  // silently added gradients onto activations.
  GpuBuf dnorm;
  GpuBuf logits, dy_ce, losses, loss_out;
  GpuBuf ids, tgt;
  cudaStream_t st = nullptr;

  void init(Model* mm, BPETokenizer* t, int seq_len) {
    m = mm; tk = t; c = mm->cfg;
    d = c.d_model; ff = c.d_ff; H = c.n_head; hd = c.head_dim();
    V = c.vocab; NL = c.n_layer; L = seq_len;
    // Anything longer than the context the model was configured with would
    // index past the per-layer attention slabs. Fail loudly instead of taking
    // an illegal memory access a few hundred steps later.
    if (L > c.ctx) {
      std::fprintf(stderr, "[init] seq_len %d exceeds model ctx %d\n", L, c.ctx);
      std::exit(4);
    }
    CUDA_CHECK(cudaStreamCreate(&st));

    int sz[NQT][2] = {{d,d},{d,d},{d,d},{d,d},{ff,d},{ff,d},{d,ff}};
    lay.resize(NL);
    for (auto& D : lay) {
      for (int j = 0; j < NQT; ++j) {
        int o = sz[j][0], in = sz[j][1];
        size_t wb = (size_t)o * in * 4;
        D.W[j].alloc(wb); D.dW[j].alloc(wb);
        D.am[j].alloc(wb); D.av[j].alloc(wb);
        D.P[j].alloc((size_t)o * (in/32) * 4);
        D.M[j].alloc((size_t)o * (in/32) * 4);
        D.scl[j].alloc((size_t)o * 4);
      }
      D.rms_att.alloc(d*4); D.rms_ffn.alloc(d*4);
      D.d_rms_att.alloc(d*4); D.d_rms_ffn.alloc(d*4);
      D.am_ra.alloc(d*4); D.av_ra.alloc(d*4);
      D.am_rf.alloc(d*4); D.av_rf.alloc(d*4);
      D.xn_att.alloc((size_t)L*d*4); D.rrms_att.alloc((size_t)L*4);
      D.q.alloc((size_t)L*d*4); D.k.alloc((size_t)L*d*4);
      D.v.alloc((size_t)L*d*4); D.o.alloc((size_t)L*d*4);
      // The attention probability slab is [L heads x L queries x L keys]. It
      // is the single largest activation buffer, so it is the one that overruns
      // first if L used for sizing drifts from the L used at run time.
      D.probs.alloc((size_t)L*H*L*4);
      D.x_mid.alloc((size_t)L*d*4);
      D.xn_ffn.alloc((size_t)L*d*4); D.rrms_ffn.alloc((size_t)L*4);
      D.gate.alloc((size_t)L*ff*4); D.up.alloc((size_t)L*ff*4);
      D.sig.alloc((size_t)L*ff*4);  D.hmid.alloc((size_t)L*ff*4);
    }

    emb_f32.alloc((size_t)V*d*4);  emb_dW.alloc((size_t)V*d*4);
    emb_am.alloc((size_t)V*d*4);   emb_av.alloc((size_t)V*d*4);
    head_f32.alloc((size_t)V*d*4); head_dW.alloc((size_t)V*d*4);
    head_am.alloc((size_t)V*d*4);  head_av.alloc((size_t)V*d*4);
    rmsf.alloc(d*4); d_rmsf.alloc(d*4); rmsf_am.alloc(d*4); rmsf_av.alloc(d*4);

    size_t LD = (size_t)L*d*4, LF = (size_t)L*ff*4;
    x.alloc(LD); dx.alloc(LD); dxb.alloc(LD); xn.alloc(LD);
    dtmp.alloc(LD); dtmp2.alloc(LD);
    rrms.alloc((size_t)L*4); d_rrms.alloc((size_t)L*4);
    dq.alloc(LD); dk.alloc(LD); dv.alloc(LD); datt.alloc(LD);
    dprobs.alloc((size_t)L*H*L*4);
    dgate.alloc(LF); dup.alloc(LF); dhmid.alloc(LF);
    dff.alloc(LD); dffb.alloc(LD); dxff.alloc(LD); dnorm.alloc(LD);
    logits.alloc((size_t)L*V*4); dy_ce.alloc((size_t)L*V*4);
    losses.alloc((size_t)L*4); loss_out.alloc(4);
    ids.alloc((size_t)(L+1)*4); tgt.alloc((size_t)L*4);
    CUDA_CHECK(cudaStreamSynchronize(st));
  }
  ~TrainerImpl() { if (st) cudaStreamDestroy(st); }
};

// ---------------------------------------------------------------------------
//  Re-derive every packed plane + scale from the fp32 masters.
// ---------------------------------------------------------------------------
static void requant_all(TrainerImpl& T) {
  int sz[NQT][2] = {{T.d,T.d},{T.d,T.d},{T.d,T.d},{T.d,T.d},{T.ff,T.d},{T.ff,T.d},{T.d,T.ff}};
  for (auto& D : T.lay)
    for (int j = 0; j < NQT; ++j)
      launch_ternarize_rows((const float*)D.W[j].ptr, (uint32_t*)D.P[j].ptr,
                            (uint32_t*)D.M[j].ptr, (float*)D.scl[j].ptr,
                            sz[j][0], sz[j][1], T.st);
  CUDA_CHECK(cudaStreamSynchronize(T.st));
}

// ---------------------------------------------------------------------------
//  One forward+backward for a single sequence. Accumulates gradients into the
//  per-layer dW / d_rms / emb_dW / head_dW buffers. If do_update is false the
//  gradient buffers are wiped afterwards (measurement pass).
// ---------------------------------------------------------------------------
static float train_one_seq(TrainerImpl& T, const Seq& sq, bool do_update, int step) {
  const int d = T.d, ff = T.ff, H = T.H, hd = T.hd, V = T.V, L = T.L;
  cudaStream_t s = T.st;

  // ---- upload the sequence ----
  // The target loop below reads sq.ids[i+1] for i in 0..L-1, so the vector must
  // hold exactly L+1 entries. A shorter one reads past the end (which is what
  // produced the CUDA illegal memory access); a longer one silently shifts every
  // target. Assert the invariant instead of hoping.
  if ((int)sq.ids.size() != L + 1) {
    std::fprintf(stderr, "[train] sequence has %zu ids, expected %d\n",
                 sq.ids.size(), L + 1);
    std::exit(5);
  }
  // Targets are the next token. Positions whose TARGET is padding are masked
  // with a -1 sentinel: k_ce_rows turns that into zero loss and zero gradient.
  // Without this the model spends most of its gradient budget on <pad>.
  int pad_id = -1;
  if (T.tk) {
    auto itp = T.tk->enc.find("<pad>");
    if (itp != T.tk->enc.end()) pad_id = itp->second;
  }
  std::vector<int> tgt(L);
  int n_mask = 0;
  for (int i = 0; i < L; ++i) {
    int t = sq.ids[i + 1];
    if (t == pad_id) { tgt[i] = -1; n_mask++; } else tgt[i] = t;
  }
  (void)n_mask;
  CUDA_CHECK(cudaMemcpyAsync(T.ids.ptr, sq.ids.data(), (L+1)*4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(T.tgt.ptr, tgt.data(), L*4, cudaMemcpyHostToDevice, s));

  // ---- zero EVERY accumulator ----
  // Three kernels use += or atomicAdd rather than assignment:
  //   k_silu_bwd   -> dgate, dup
  //   k_attn_bwd   -> dq, dk, dv
  //   k_rms_bwd    -> d_rms_*, d_rmsf
  // and the dW collectors accumulate. Missing any one of these makes gradients
  // grow without bound across steps; that showed up as loss -> NaN by step 26.
  for (auto& D : T.lay) {
    for (int j = 0; j < NQT; ++j) launch_zero((float*)D.dW[j].ptr, (int)(D.dW[j].bytes/4), s);
    launch_zero((float*)D.d_rms_att.ptr, d, s);
    launch_zero((float*)D.d_rms_ffn.ptr, d, s);
  }
  launch_zero((float*)T.emb_dW.ptr, V*d, s);
  launch_zero((float*)T.head_dW.ptr, V*d, s);
  launch_zero((float*)T.d_rmsf.ptr, d, s);
  launch_zero((float*)T.dq.ptr, L*d, s);
  launch_zero((float*)T.dk.ptr, L*d, s);
  launch_zero((float*)T.dv.ptr, L*d, s);
  launch_zero((float*)T.dgate.ptr, L*ff, s);
  launch_zero((float*)T.dup.ptr, L*ff, s);

  // ============================== FORWARD ==============================
  embed_fwd32((const float*)T.emb_f32.ptr, (const int*)T.ids.ptr, (float*)T.x.ptr, L, d, s);
  copy_f32((float*)T.dx.ptr, (const float*)T.x.ptr, L*d, s);   // running residual

  int sz[NQT][2] = {{d,d},{d,d},{d,d},{d,d},{ff,d},{ff,d},{d,ff}};

  for (int l = 0; l < T.NL; ++l) {
    auto& D = T.lay[l];

    // ---- attention block ----
    fw_rmsnorm((const float*)T.dx.ptr, (const float*)D.rms_att.ptr,
               (float*)D.xn_att.ptr, (float*)D.rrms_att.ptr, L, d, 1e-6f, s);

    launch_tgemm_fwd((const float*)D.xn_att.ptr, (const uint32_t*)D.P[0].ptr,
                     (const uint32_t*)D.M[0].ptr, (const float*)D.scl[0].ptr,
                     (float*)D.q.ptr, L, d, d, s);
    launch_tgemm_fwd((const float*)D.xn_att.ptr, (const uint32_t*)D.P[1].ptr,
                     (const uint32_t*)D.M[1].ptr, (const float*)D.scl[1].ptr,
                     (float*)D.k.ptr, L, d, d, s);
    launch_tgemm_fwd((const float*)D.xn_att.ptr, (const uint32_t*)D.P[2].ptr,
                     (const uint32_t*)D.M[2].ptr, (const float*)D.scl[2].ptr,
                     (float*)D.v.ptr, L, d, d, s);

    fw_rope((float*)D.q.ptr, (float*)D.k.ptr, L, H, hd, 10000.f, s);

    fw_attn((const float*)D.q.ptr, (const float*)D.k.ptr, (const float*)D.v.ptr,
            (float*)D.o.ptr, (float*)D.probs.ptr, L, H, hd, 1.0f/sqrtf((float)hd), s);

    launch_tgemm_fwd((const float*)D.o.ptr, (const uint32_t*)D.P[3].ptr,
                     (const uint32_t*)D.M[3].ptr, (const float*)D.scl[3].ptr,
                     (float*)T.dtmp.ptr, L, d, d, s);

    add_inplace32((float*)T.dx.ptr, (const float*)T.dtmp.ptr, L*d, s);
    copy_f32((float*)D.x_mid.ptr, (const float*)T.dx.ptr, L*d, s);   // residual entry

    // ---- ffn block ----
    fw_rmsnorm((const float*)T.dx.ptr, (const float*)D.rms_ffn.ptr,
               (float*)D.xn_ffn.ptr, (float*)D.rrms_ffn.ptr, L, d, 1e-6f, s);

    launch_tgemm_fwd((const float*)D.xn_ffn.ptr, (const uint32_t*)D.P[4].ptr,
                     (const uint32_t*)D.M[4].ptr, (const float*)D.scl[4].ptr,
                     (float*)D.gate.ptr, L, ff, d, s);
    launch_tgemm_fwd((const float*)D.xn_ffn.ptr, (const uint32_t*)D.P[5].ptr,
                     (const uint32_t*)D.M[5].ptr, (const float*)D.scl[5].ptr,
                     (float*)D.up.ptr, L, ff, d, s);
    fw_silu((const float*)D.gate.ptr, (const float*)D.up.ptr,
            (float*)D.hmid.ptr, (float*)D.sig.ptr, L*ff, s);
    launch_tgemm_fwd((const float*)D.hmid.ptr, (const uint32_t*)D.P[6].ptr,
                     (const uint32_t*)D.M[6].ptr, (const float*)D.scl[6].ptr,
                     (float*)T.dtmp.ptr, L, d, ff, s);

    add_inplace32((float*)T.dx.ptr, (const float*)T.dtmp.ptr, L*d, s);
  }

  // ---- final norm + head ----
  fw_rmsnorm((const float*)T.dx.ptr, (const float*)T.rmsf.ptr,
             (float*)T.xn.ptr, (float*)T.rrms.ptr, L, d, 1e-6f, s);
  launch_dgemm_fwd((const float*)T.xn.ptr, (const float*)T.head_f32.ptr,
                   (float*)T.logits.ptr, L, V, d, s);

  // ---- loss ----
  launch_ce_rows((const float*)T.logits.ptr, (const int*)T.tgt.ptr,
                 (float*)T.dy_ce.ptr, (float*)T.losses.ptr, L, V, s);
  // Masked rows contribute 0 to the sum, so dividing by the number of UNMASKED
  // rows (not by L) is what makes the reported loss comparable across batches.
  int n_eff = 0;
  if (do_update) { for (int i = 0; i < L; ++i) if (tgt[i] >= 0) n_eff++; }
  launch_mean_reduce((const float*)T.losses.ptr, (float*)T.loss_out.ptr,
                     n_eff > 0 ? n_eff : 1, s);
  float loss = 0;
  CUDA_CHECK(cudaMemcpyAsync(&loss, T.loss_out.ptr, 4, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  if (!do_update) return loss;

  // ============================== BACKWARD =============================
  // dlogits is already in dy_ce.
  // head is DENSE: dW[v][j] = sum_m dy[m][v] * xn[m][j]
  launch_dgemm_dW((const float*)T.dy_ce.ptr, (const float*)T.xn.ptr,
                  (float*)T.head_dW.ptr, L, V, d, s);
  //               dxn[m][j] = sum_v dy[m][v] * W[v][j]
  launch_dgemm_dx((const float*)T.dy_ce.ptr, (const float*)T.head_f32.ptr,
                  (float*)T.dtmp.ptr, L, V, d, s);

  launch_zero((float*)T.dnorm.ptr, L*d, s);
  bw_rmsnorm((const float*)T.dx.ptr, (const float*)T.rmsf.ptr, (const float*)T.dtmp.ptr,
             (const float*)T.rrms.ptr, (float*)T.dnorm.ptr, (float*)T.d_rmsf.ptr, L, d, s);
  copy_f32((float*)T.dx.ptr, (const float*)T.dnorm.ptr, L*d, s);

  // ---- per layer, in reverse ----
  for (int l = T.NL - 1; l >= 0; --l) {
    auto& D = T.lay[l];

    // ffn: residual split. dff goes into the down-projection path.
    copy_f32((float*)T.dff.ptr, (const float*)T.dx.ptr, L*d, s);
    launch_tgemm_dW((const float*)T.dff.ptr, (const float*)D.hmid.ptr,
                    (const float*)D.scl[6].ptr, (float*)D.dW[6].ptr, L, d, ff, s);
    launch_tgemm_dx((const float*)T.dff.ptr, (const uint32_t*)D.P[6].ptr,
                    (const uint32_t*)D.M[6].ptr, (const float*)D.scl[6].ptr,
                    (float*)T.dhmid.ptr, L, d, ff, s);
    bw_silu((const float*)D.gate.ptr, (const float*)D.up.ptr, (const float*)D.sig.ptr,
            (const float*)T.dhmid.ptr, (float*)T.dgate.ptr, (float*)T.dup.ptr, L*ff, s);
    launch_tgemm_dW((const float*)T.dgate.ptr, (const float*)D.xn_ffn.ptr,
                    (const float*)D.scl[4].ptr, (float*)D.dW[4].ptr, L, ff, d, s);
    launch_tgemm_dW((const float*)T.dup.ptr, (const float*)D.xn_ffn.ptr,
                    (const float*)D.scl[5].ptr, (float*)D.dW[5].ptr, L, ff, d, s);
    // dxn_ffn = contribution from gate and up
    launch_tgemm_dx((const float*)T.dgate.ptr, (const uint32_t*)D.P[4].ptr,
                    (const uint32_t*)D.M[4].ptr, (const float*)D.scl[4].ptr,
                    (float*)T.dtmp.ptr, L, ff, d, s);
    launch_tgemm_dx((const float*)T.dup.ptr, (const uint32_t*)D.P[5].ptr,
                    (const uint32_t*)D.M[5].ptr, (const float*)D.scl[5].ptr,
                    (float*)T.dtmp2.ptr, L, ff, d, s);
    add_inplace32((float*)T.dtmp.ptr, (const float*)T.dtmp2.ptr, L*d, s);
    launch_zero((float*)T.dnorm.ptr, L*d, s);
    bw_rmsnorm((const float*)D.x_mid.ptr, (const float*)D.rms_ffn.ptr, (const float*)T.dtmp.ptr,
               (const float*)D.rrms_ffn.ptr, (float*)T.dnorm.ptr, (float*)D.d_rms_ffn.ptr, L, d, s);
    add_inplace32((float*)T.dx.ptr, (const float*)T.dnorm.ptr, L*d, s);

    // attention: residual split
    copy_f32((float*)T.datt.ptr, (const float*)T.dx.ptr, L*d, s);
    launch_tgemm_dW((const float*)T.datt.ptr, (const float*)D.o.ptr,
                    (const float*)D.scl[3].ptr, (float*)D.dW[3].ptr, L, d, d, s);
    launch_tgemm_dx((const float*)T.datt.ptr, (const uint32_t*)D.P[3].ptr,
                    (const uint32_t*)D.M[3].ptr, (const float*)D.scl[3].ptr,
                    (float*)T.dtmp.ptr, L, d, d, s);   // dO
    bw_attn((const float*)D.q.ptr, (const float*)D.k.ptr, (const float*)D.v.ptr,
            (const float*)D.o.ptr, (const float*)D.probs.ptr, (const float*)T.dtmp.ptr,
            (float*)T.dprobs.ptr, (float*)T.dq.ptr, (float*)T.dk.ptr, (float*)T.dv.ptr,
            L, H, hd, 1.0f/sqrtf((float)hd), s);
    bw_rope((float*)T.dq.ptr, (float*)T.dk.ptr, L, H, hd, 10000.f, s);

    launch_tgemm_dW((const float*)T.dq.ptr, (const float*)D.xn_att.ptr,
                    (const float*)D.scl[0].ptr, (float*)D.dW[0].ptr, L, d, d, s);
    launch_tgemm_dW((const float*)T.dk.ptr, (const float*)D.xn_att.ptr,
                    (const float*)D.scl[1].ptr, (float*)D.dW[1].ptr, L, d, d, s);
    launch_tgemm_dW((const float*)T.dv.ptr, (const float*)D.xn_att.ptr,
                    (const float*)D.scl[2].ptr, (float*)D.dW[2].ptr, L, d, d, s);

    launch_tgemm_dx((const float*)T.dq.ptr, (const uint32_t*)D.P[0].ptr,
                    (const uint32_t*)D.M[0].ptr, (const float*)D.scl[0].ptr,
                    (float*)T.dtmp.ptr, L, d, d, s);
    launch_tgemm_dx((const float*)T.dk.ptr, (const uint32_t*)D.P[1].ptr,
                    (const uint32_t*)D.M[1].ptr, (const float*)D.scl[1].ptr,
                    (float*)T.dtmp2.ptr, L, d, d, s);
    add_inplace32((float*)T.dtmp.ptr, (const float*)T.dtmp2.ptr, L*d, s);
    launch_tgemm_dx((const float*)T.dv.ptr, (const uint32_t*)D.P[2].ptr,
                    (const uint32_t*)D.M[2].ptr, (const float*)D.scl[2].ptr,
                    (float*)T.dtmp2.ptr, L, d, d, s);
    add_inplace32((float*)T.dtmp.ptr, (const float*)T.dtmp2.ptr, L*d, s);
    launch_zero((float*)T.dnorm.ptr, L*d, s);
    bw_rmsnorm((const float*)T.dx.ptr, (const float*)D.rms_att.ptr, (const float*)T.dtmp.ptr,
               (const float*)D.rrms_att.ptr, (float*)T.dnorm.ptr, (float*)D.d_rms_att.ptr, L, d, s);
    add_inplace32((float*)T.dx.ptr, (const float*)T.dnorm.ptr, L*d, s);
  }

  // ---- embedding gather backward ----
  launch_embed_bwd32((float*)T.emb_dW.ptr, (const int*)T.ids.ptr, (const float*)T.dx.ptr, L, d, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  return loss;
}

// ---------------------------------------------------------------------------
//  AdamW over every master tensor + optional re-ternarisation.
// ---------------------------------------------------------------------------
// Diagnostics: count non-finite entries in the gradient buffers BEFORE the
// clip pass, because clipping silently rewrites NaN to 0 and hides the origin.
static void probe_nans(TrainerImpl& T, const char* tag) {
  static int* d_cnt = nullptr;
  if (!d_cnt) CUDA_CHECK(cudaMalloc(&d_cnt, 4));
  auto& s = T.st;
  auto count = [&](const void* p, long n, const char* name) {
    int zero = 0;
    CUDA_CHECK(cudaMemcpyAsync(d_cnt, &zero, 4, cudaMemcpyHostToDevice, s));
    launch_nan_count((const float*)p, n, d_cnt, s);
    int c = 0;
    CUDA_CHECK(cudaMemcpyAsync(&c, d_cnt, 4, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    if (c) std::printf("    [%s] %s: %d non-finite of %ld\n", tag, name, c, n);
    return c;
  };
  int total = 0;
  total += count(T.emb_dW.ptr, (long)T.V*T.d, "emb_dW");
  total += count(T.head_dW.ptr, (long)T.V*T.d, "head_dW");
  total += count(T.dq.ptr, (long)T.L*T.d, "dq");
  total += count(T.dk.ptr, (long)T.L*T.d, "dk");
  total += count(T.dv.ptr, (long)T.L*T.d, "dv");
  total += count(T.dgate.ptr, (long)T.L*T.ff, "dgate");
  total += count(T.dup.ptr, (long)T.L*T.ff, "dup");
  for (int l = 0; l < T.NL; ++l)
    for (int j = 0; j < NQT; ++j) {
      char nm[32]; std::snprintf(nm, sizeof(nm), "L%d.dW%d", l, j);
      total += count(T.lay[l].dW[j].ptr, (long)T.lay[l].dW[j].bytes/4, nm);
    }
  if (!total) std::printf("    [%s] no non-finite gradients\n", tag);
}

static void apply_update(TrainerImpl& T, const TrainCfg& cfg, int step) {
  auto& s = T.st;
  const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  // Bias correction with a +1: apply_update is called with the 0-based step
  // index, and 1 - b^0 == 0 would make Adam divide by zero on the very first
  // update (observed as loss -> NaN within 20 steps).
  const float t = (float)(step + 1);
  const float bc1 = 1.f - std::pow(b1, t);
  const float bc2 = 1.f - std::pow(b2, t);
  // linear warmup then cosine decay: with a 3k-step budget a flat LR either
  // diverges early or crawls late.
  float lr = cfg.lr;
  if (step < 100) lr = cfg.lr * (step + 1) / 100.0f;
  else lr = cfg.lr * 0.5f * (1.f + cosf(3.14159265f * (step - 100) / std::max(1, cfg.steps - 100)));

  int sz[NQT][2] = {{T.d,T.d},{T.d,T.d},{T.d,T.d},{T.d,T.d},{T.ff,T.d},{T.ff,T.d},{T.d,T.ff}};
  for (auto& D : T.lay) {
    for (int j = 0; j < NQT; ++j) {
      long n = (long)sz[j][0] * sz[j][1];
      // Clip before Adam: with fp32 masters and STE gradients from a discrete
      // forward pass, an early outlier weight of large magnitude produces a
      // gradient big enough to poison every moment estimate at once.
      launch_clip_grad((float*)D.dW[j].ptr, n, cfg.clip, s);
      launch_adamw2((float*)D.W[j].ptr, (const float*)D.dW[j].ptr,
                    (float*)D.am[j].ptr, (float*)D.av[j].ptr, n,
                    lr, b1, b2, eps, cfg.wd, bc1, bc2, s);
    }
    launch_adamw2((float*)D.rms_att.ptr, (const float*)D.d_rms_att.ptr,
                  (float*)D.am_ra.ptr, (float*)D.av_ra.ptr, T.d,
                  lr, b1, b2, eps, 0.f, bc1, bc2, s);
    launch_adamw2((float*)D.rms_ffn.ptr, (const float*)D.d_rms_ffn.ptr,
                  (float*)D.am_rf.ptr, (float*)D.av_rf.ptr, T.d,
                  lr, b1, b2, eps, 0.f, bc1, bc2, s);
  }
  launch_clip_grad((float*)T.emb_dW.ptr, (long)T.V*T.d, cfg.clip, s);
  launch_adamw2((float*)T.emb_f32.ptr, (const float*)T.emb_dW.ptr,
                (float*)T.emb_am.ptr, (float*)T.emb_av.ptr, (long)T.V*T.d,
                lr, b1, b2, eps, 0.f, bc1, bc2, s);
  launch_clip_grad((float*)T.head_dW.ptr, (long)T.V*T.d, cfg.clip, s);
  launch_adamw2((float*)T.head_f32.ptr, (const float*)T.head_dW.ptr,
                (float*)T.head_am.ptr, (float*)T.head_av.ptr, (long)T.V*T.d,
                lr, b1, b2, eps, cfg.wd, bc1, bc2, s);
  launch_adamw2((float*)T.rmsf.ptr, (const float*)T.d_rmsf.ptr,
                (float*)T.rmsf_am.ptr, (float*)T.rmsf_av.ptr, T.d,
                lr, b1, b2, eps, 0.f, bc1, bc2, s);
}

// ---------------------------------------------------------------------------
//  Seed the fp32 masters from the model's host-side ternary planes, so the
//  trainer starts from exactly the state the inference forward would use.
// ---------------------------------------------------------------------------
static void seed_masters_from_model(TrainerImpl& T, Model& m) {
  const int d = T.d, ff = T.ff, V = T.V;
  cudaStream_t s = T.st;
  int sz[NQT][2] = {{d,d},{d,d},{d,d},{d,d},{ff,d},{ff,d},{d,ff}};
  for (int l = 0; l < m.cfg.n_layer; ++l) {
    auto& ML = m.layers[l];
    auto& DL = T.lay[l];
    TernLinear* src[7] = {&ML.wq,&ML.wk,&ML.wv,&ML.wo,&ML.w_gate,&ML.w_up,&ML.w_down};
    for (int j = 0; j < 7; ++j) {
      int o = sz[j][0], in = sz[j][1];
      std::vector<float> master((size_t)o * in);
      for (int n = 0; n < o; ++n) {
        float sc = f16_to_f32(src[j]->scale[n]);
        for (int k = 0; k < in; ++k) {
          size_t wi = (size_t)n * (in/32) + (k>>5);
          uint32_t bit = 1u << (k & 31);
          int w = (int)((src[j]->W.P[wi] & bit) != 0) - (int)((src[j]->W.M[wi] & bit) != 0);
          master[(size_t)n*in + k] = (float)w * sc;
        }
      }
      CUDA_CHECK(cudaMemcpyAsync(DL.W[j].ptr, master.data(), master.size()*4,
                                 cudaMemcpyHostToDevice, s));
    }
    std::vector<float> ra(d, 1.f), rf(d, 1.f);
    CUDA_CHECK(cudaMemcpyAsync(DL.rms_att.ptr, ra.data(), d*4, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(DL.rms_ffn.ptr, rf.data(), d*4, cudaMemcpyHostToDevice, s));
  }
  std::vector<float> emb((size_t)V*d), headw((size_t)V*d);
  for (size_t i = 0; i < emb.size(); ++i)  emb[i]   = f16_to_f32(m.token_emb[i]);
  for (size_t i = 0; i < headw.size(); ++i) headw[i] = f16_to_f32(m.lm_head[i]);
  CUDA_CHECK(cudaMemcpyAsync(T.emb_f32.ptr,  emb.data(),  emb.size()*4,  cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(T.head_f32.ptr, headw.data(), headw.size()*4, cudaMemcpyHostToDevice, s));
  std::vector<float> rf(d, 1.f);
  CUDA_CHECK(cudaMemcpyAsync(T.rmsf.ptr, rf.data(), d*4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
}

// ---------------------------------------------------------------------------
//  Push the trained masters back into the host-side Model so the inference
//  forward (and save_checkpoint) sees what training produced.
// ---------------------------------------------------------------------------
static void export_to_model(TrainerImpl& T, Model& m) {
  const int d = T.d, ff = T.ff, V = T.V;
  cudaStream_t s = T.st;
  int sz[NQT][2] = {{d,d},{d,d},{d,d},{d,d},{ff,d},{ff,d},{d,ff}};
  for (int l = 0; l < m.cfg.n_layer; ++l) {
    auto& ML = m.layers[l];
    auto& DL = T.lay[l];
    TernLinear* dsts[7] = {&ML.wq,&ML.wk,&ML.wv,&ML.wo,&ML.w_gate,&ML.w_up,&ML.w_down};
    for (int j = 0; j < 7; ++j) {
      int o = sz[j][0], in = sz[j][1];
      std::vector<float> sc(o);
      CUDA_CHECK(cudaMemcpyAsync(sc.data(), DL.scl[j].ptr, o*4, cudaMemcpyDeviceToHost, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
      std::vector<uint32_t> Pp((size_t)o*(in/32)), Mp((size_t)o*(in/32));
      CUDA_CHECK(cudaMemcpyAsync(Pp.data(), DL.P[j].ptr, Pp.size()*4, cudaMemcpyDeviceToHost, s));
      CUDA_CHECK(cudaMemcpyAsync(Mp.data(), DL.M[j].ptr, Mp.size()*4, cudaMemcpyDeviceToHost, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
      dsts[j]->W.P = Pp; dsts[j]->W.M = Mp;
      dsts[j]->scale.resize(o);
      for (int n = 0; n < o; ++n) dsts[j]->scale[n] = f32_to_f16(sc[n]);
    }
    std::vector<float> ra(d), rf(d);
    CUDA_CHECK(cudaMemcpyAsync(ra.data(), DL.rms_att.ptr, d*4, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(rf.data(), DL.rms_ffn.ptr, d*4, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    for (int i = 0; i < d; ++i) { ML.rms_att[i] = f32_to_f16(ra[i]); ML.rms_ffn[i] = f32_to_f16(rf[i]); }
  }
  std::vector<float> emb((size_t)V*d), headw((size_t)V*d), rf(d);
  CUDA_CHECK(cudaMemcpyAsync(emb.data(),  T.emb_f32.ptr,  emb.size()*4,  cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaMemcpyAsync(headw.data(), T.head_f32.ptr, headw.size()*4, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaMemcpyAsync(rf.data(), T.rmsf.ptr, d*4, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  for (size_t i = 0; i < emb.size(); ++i)   m.token_emb[i] = f32_to_f16(emb[i]);
  for (size_t i = 0; i < headw.size(); ++i) m.lm_head[i]  = f32_to_f16(headw[i]);
  for (int i = 0; i < d; ++i) m.rms_final[i] = f32_to_f16(rf[i]);
}

int trainer_forward_probe(Model& m, BPETokenizer& tk, const std::vector<Sample>& samples) {
  auto seqs = pack_samples(samples, tk, tool_system_prompt(), 64);
  if (seqs.empty()) { std::printf("  no seqs\n"); return 1; }
  TrainerImpl T;
  T.init(&m, &tk, 64);
  seed_masters_from_model(T, m);
  requant_all(T);
  for (int rep = 0; rep < 8; ++rep) {
    float loss = train_one_seq(T, seqs[0], false, 0);
    std::printf("  fwd rep %d  loss %.6f  %s\n", rep, loss, (loss == loss) ? "finite" : "NaN");
  }
  std::printf("  --- with backward, no update ---\n");
  for (int rep = 0; rep < 4; ++rep) {
    float loss = train_one_seq(T, seqs[0], true, 0);
    std::printf("  fwd+bwd rep %d  loss %.6f  %s\n", rep, loss, (loss == loss) ? "finite" : "NaN");
  }
  return 0;
}

// ---------------------------------------------------------------------------
//  Hook for the alignment probe: run the TRAINING forward on a full sequence
//  and return the logits at the final position. This is the function the
//  optimiser actually differentiates. model.cpp has a SEPARATE inference
//  forward with different kernels and fp16 activations; if the two disagree,
//  training and inference are optimising/evaluating different functions.
// ---------------------------------------------------------------------------
int train_forward_logits(Model& m, const std::vector<int>& ids,
                         std::vector<float>& out_logits) {
  TrainerImpl T;
  T.init(&m, nullptr, (int)ids.size());
  seed_masters_from_model(T, m);
  requant_all(T);

  cudaStream_t s = T.st;
  const int L = (int)ids.size(), d = T.d, V = T.V, ff = T.ff, H = T.H, hd = T.hd;

  std::vector<int> tgt(L, 0);
  CUDA_CHECK(cudaMemcpyAsync(T.ids.ptr, ids.data(), L*4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(T.tgt.ptr, tgt.data(), L*4, cudaMemcpyHostToDevice, s));

  embed_fwd32((const float*)T.emb_f32.ptr, (const int*)T.ids.ptr, (float*)T.x.ptr, L, d, s);
  copy_f32((float*)T.dx.ptr, (const float*)T.x.ptr, L*d, s);

  for (int l = 0; l < T.NL; ++l) {
    auto& D = T.lay[l];
    fw_rmsnorm((const float*)T.dx.ptr, (const float*)D.rms_att.ptr,
               (float*)D.xn_att.ptr, (float*)D.rrms_att.ptr, L, d, 1e-6f, s);
    launch_tgemm_fwd((const float*)D.xn_att.ptr, (const uint32_t*)D.P[0].ptr,
                     (const uint32_t*)D.M[0].ptr, (const float*)D.scl[0].ptr,
                     (float*)D.q.ptr, L, d, d, s);
    launch_tgemm_fwd((const float*)D.xn_att.ptr, (const uint32_t*)D.P[1].ptr,
                     (const uint32_t*)D.M[1].ptr, (const float*)D.scl[1].ptr,
                     (float*)D.k.ptr, L, d, d, s);
    launch_tgemm_fwd((const float*)D.xn_att.ptr, (const uint32_t*)D.P[2].ptr,
                     (const uint32_t*)D.M[2].ptr, (const float*)D.scl[2].ptr,
                     (float*)D.v.ptr, L, d, d, s);
    fw_rope((float*)D.q.ptr, (float*)D.k.ptr, L, H, hd, 10000.f, s);
    fw_attn((const float*)D.q.ptr, (const float*)D.k.ptr, (const float*)D.v.ptr,
            (float*)D.o.ptr, (float*)D.probs.ptr, L, H, hd, 1.0f/sqrtf((float)hd), s);
    launch_tgemm_fwd((const float*)D.o.ptr, (const uint32_t*)D.P[3].ptr,
                     (const uint32_t*)D.M[3].ptr, (const float*)D.scl[3].ptr,
                     (float*)T.dtmp.ptr, L, d, d, s);
    add_inplace32((float*)T.dx.ptr, (const float*)T.dtmp.ptr, L*d, s);
    copy_f32((float*)D.x_mid.ptr, (const float*)T.dx.ptr, L*d, s);

    fw_rmsnorm((const float*)T.dx.ptr, (const float*)D.rms_ffn.ptr,
               (float*)D.xn_ffn.ptr, (float*)D.rrms_ffn.ptr, L, d, 1e-6f, s);
    launch_tgemm_fwd((const float*)D.xn_ffn.ptr, (const uint32_t*)D.P[4].ptr,
                     (const uint32_t*)D.M[4].ptr, (const float*)D.scl[4].ptr,
                     (float*)D.gate.ptr, L, ff, d, s);
    launch_tgemm_fwd((const float*)D.xn_ffn.ptr, (const uint32_t*)D.P[5].ptr,
                     (const uint32_t*)D.M[5].ptr, (const float*)D.scl[5].ptr,
                     (float*)D.up.ptr, L, ff, d, s);
    fw_silu((const float*)D.gate.ptr, (const float*)D.up.ptr,
            (float*)D.hmid.ptr, (float*)D.sig.ptr, L*ff, s);
    launch_tgemm_fwd((const float*)D.hmid.ptr, (const uint32_t*)D.P[6].ptr,
                     (const uint32_t*)D.M[6].ptr, (const float*)D.scl[6].ptr,
                     (float*)T.dtmp.ptr, L, d, ff, s);
    add_inplace32((float*)T.dx.ptr, (const float*)T.dtmp.ptr, L*d, s);
  }

  fw_rmsnorm((const float*)T.dx.ptr, (const float*)T.rmsf.ptr,
             (float*)T.xn.ptr, (float*)T.rrms.ptr, L, d, 1e-6f, s);
  // logits for the LAST position only
  const float* lastp = (const float*)T.xn.ptr + (size_t)(L-1)*d;
  launch_dgemm_fwd(lastp, (const float*)T.head_f32.ptr,
                   (float*)T.logits.ptr, 1, V, d, s);

  out_logits.resize(V);
  CUDA_CHECK(cudaMemcpyAsync(out_logits.data(), T.logits.ptr, V*4, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  return 0;
}

int full_train_run(int argc, char** argv) {
  TrainCfg cfg;
  std::string load_path, cfg_out = "build\\ckpt\\canticle.bin";
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--steps" && i+1 < argc) cfg.steps = atoi(argv[i+1]);
    if (a == "--seq"   && i+1 < argc) cfg.seq_len = atoi(argv[i+1]);
    if (a == "--lr"    && i+1 < argc) cfg.lr = (float)atof(argv[i+1]);
    if (a == "--rq"    && i+1 < argc) cfg.requant_every = atoi(argv[i+1]);
    if (a == "--data"  && i+1 < argc) cfg.n_samples = atoi(argv[i+1]);
    if (a == "--out"   && i+1 < argc) cfg_out = argv[i+1];
    if (a == "--load"  && i+1 < argc) load_path = argv[i+1];
    if (a == "--d"     && i+1 < argc) cfg.d_model = atoi(argv[i+1]);
    if (a == "--layers"&& i+1 < argc) cfg.n_layer = atoi(argv[i+1]);
    if (a == "--tiny") { cfg.d_model = 128; cfg.n_layer = 2; cfg.n_head = 2;
                         cfg.d_ff = 384; cfg.vocab = 1024; cfg.seq_len = 64;
                         cfg.n_samples = 600; }
  }

  std::printf("=== CANTICLE full-model ternary training ===\n");
  std::printf("steps=%d seq=%d lr=%.1e rq=%d samples=%d\n",
              cfg.steps, cfg.seq_len, cfg.lr, cfg.requant_every, cfg.n_samples);

  // ---- tokenizer ----
  auto samples = synth_tool_data(cfg.n_samples, cfg.seed);
  std::string corpus;
  for (auto& s : samples) corpus += "<|user|>\n" + s.prompt + "\n<|assistant|>\n" + s.reply + "\n";
  corpus += tool_system_prompt();

  BPETokenizer tk;
  Model m;
  if (!load_path.empty() && load_checkpoint(load_path, m, &tk)) {
    std::printf("[load] %s\n", load_path.c_str());
  } else {
    double t0 = now_ms();
    tk.train(corpus, cfg.vocab);
    std::printf("[tokenizer] vocab=%zu in %.0f ms\n", tk.vocab.size(), now_ms()-t0);
    Config c;
    c.vocab = (int)tk.vocab.size();
    c.d_model = cfg.d_model; c.n_layer = cfg.n_layer; c.n_head = cfg.n_head;
    c.d_ff = cfg.d_ff; c.ctx = cfg.ctx;
    m.init(c);
    std::mt19937 rng(cfg.seed);
    std::normal_distribution<float> nd(0.f, 0.04f);
    for (auto& L : m.layers) {
      auto mk = [&](TernLinear& t) {
        for (int n = 0; n < t.out; ++n) {
          double as = 0;
          std::vector<float> w(t.in);
          for (int k = 0; k < t.in; ++k) { w[k] = nd(rng); as += std::fabs(w[k]); }
          float thr = (float)(0.7 * as / t.in);
          double keep = 0; int nk = 0;
          for (int k = 0; k < t.in; ++k) {
            uint32_t bit = 1u << (k & 31); size_t wi = (size_t)n*(t.in/32) + (k>>5);
            if (w[k] >  thr) { t.W.P[wi] |= bit; keep += std::fabs(w[k]); nk++; }
            if (w[k] < -thr) { t.W.M[wi] |= bit; keep += std::fabs(w[k]); nk++; }
          }
          t.scale[n] = f32_to_f16(nk ? (float)(keep/nk) : 1e-3f);
        }
      };
      mk(L.wq); mk(L.wk); mk(L.wv); mk(L.wo); mk(L.w_gate); mk(L.w_up); mk(L.w_down);
    }
    for (auto& v : m.token_emb) v = f32_to_f16(nd(rng));
    for (auto& v : m.lm_head)   v = f32_to_f16(nd(rng));
    for (auto& L : m.layers) {
      for (auto& v : L.rms_att) v = f32_to_f16(1.f);
      for (auto& v : L.rms_ffn) v = f32_to_f16(1.f);
    }
    for (auto& v : m.rms_final) v = f32_to_f16(1.f);
  }

  std::printf("[model] vocab=%d d=%d L=%d H=%d ff=%d  ternary=%.2f MiB\n",
              m.cfg.vocab, m.cfg.d_model, m.cfg.n_layer, m.cfg.n_head, m.cfg.d_ff,
              m.ternary_weight_bytes()/1048576.0);

  auto seqs = pack_samples(samples, tk, tool_system_prompt(), cfg.seq_len);
  if (seqs.empty()) { std::fprintf(stderr, "no training sequences\n"); return 2; }
  std::printf("[data] %zu sequences x %d tokens\n", seqs.size(), cfg.seq_len);
  // A single out-of-range target makes k_ce_rows index outside the logit row,
  // which reads garbage, overflows the softmax and turns the whole backward
  // pass into NaN. Assert the range here rather than debugging it downstream.
  {
    int lo = 1 << 30, hi = -(1 << 30);
    int bad = 0;
    for (auto& s : seqs) for (int id : s.ids) {
      lo = std::min(lo, id); hi = std::max(hi, id);
      if (id < 0 || id >= m.cfg.vocab) bad++;
    }
    std::printf("[data] id range = [%d, %d]  vocab=%d  out-of-range=%d\n",
                lo, hi, m.cfg.vocab, bad);
    if (bad) {
      std::printf("[ABORT] tokenizer produced ids outside [0,%d). ", m.cfg.vocab);
      std::printf("Special tokens in the template were not registered by train().\n");
      return 3;
    }
  }

  TrainerImpl T;
  T.init(&m, &tk, cfg.seq_len);
  seed_masters_from_model(T, m);
  requant_all(T);

  // ---- training loop ----
  double t0 = now_ms();
  float first = 0, last = 0;
  std::vector<float> recent;
  for (int step = 0; step < cfg.steps; ++step) {
    const Seq& sq = seqs[step % seqs.size()];
    bool do_upd = (step % cfg.requant_every) != 0 || true;   // always update
    float loss = train_one_seq(T, sq, true, step);
    apply_update(T, cfg, step);
    if ((step + 1) % cfg.requant_every == 0) requant_all(T);
    if (step == 0) first = loss;
    last = loss;
    recent.push_back(loss);
    if (recent.size() > 50) recent.erase(recent.begin());
    if (step % cfg.log_every == 0 || step == cfg.steps - 1 || !(loss == loss)) {
      double avg = 0; int bad = 0;
      for (float v : recent) { if (v != v) bad++; else avg += v; }
      avg /= std::max<size_t>(1, recent.size() - bad);
      // health probe: peak |master| across the ternary layers and the head
      float wmax = 0.f, hmax = 0.f;
      {
        std::vector<float> tmp(4096);
        for (auto& D : T.lay) {
          CUDA_CHECK(cudaMemcpy(tmp.data(), D.W[0].ptr, 4096*4, cudaMemcpyDeviceToHost));
          for (float v : tmp) { if (v != v) { wmax = 1e30f; break; } wmax = std::max(wmax, std::fabs(v)); }
        }
        CUDA_CHECK(cudaMemcpy(tmp.data(), T.head_f32.ptr, 4096*4, cudaMemcpyDeviceToHost));
        for (float v : tmp) { if (v != v) { hmax = 1e30f; break; } hmax = std::max(hmax, std::fabs(v)); }
      }
      std::printf("  step %5d/%d  loss %.4f  avg %.4f  |W|max %.4f  |head|max %.4f  %.0f ms\n",
                  step+1, cfg.steps, loss, avg, wmax, hmax, now_ms()-t0);
      std::fflush(stdout);
    }
    (void)do_upd;
  }
  double t1 = now_ms();
  std::printf("\ntrain: %.1f s  (%.2f steps/s)\n", (t1-t0)/1000.0, cfg.steps*1000.0/(t1-t0));
  std::printf("loss: %.4f -> %.4f\n", first, last);

  // ---- export + save ----
  export_to_model(T, m);
  if (!cfg_out.empty()) {
    namespace fs = std::filesystem;
    fs::path p(cfg_out);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    if (save_checkpoint(cfg_out, m, &tk)) std::printf("[saved] %s\n", cfg_out.c_str());
    else std::printf("[FAIL] could not save %s\n", cfg_out.c_str());
  }
  return 0;
}

} // namespace canticle

// Global forwarder: main.cpp dispatches all subcommands through plain extern
// symbols, so the namespace-qualified entry point needs a thin wrapper.
int full_train_run(int argc, char** argv) {
  return canticle::full_train_run(argc, argv);
}

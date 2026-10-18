
// ============================================================================
//  CANTICLE model: quantisation, device residency, forward pass.
// ============================================================================
#include "canticle/model.h"
#include "canticle/kernels.h"
#include "canticle/ops.h"
#include <cmath>
#include <algorithm>
#include <random>

namespace canticle {

// ------------------------------------------------------- master weight size
size_t MasterWeights::numel() const {
  size_t n = token_emb.size() + rms_final.size() + lm_head.size();
  for (auto& l : layers)
    n += l.wq.size()+l.wk.size()+l.wv.size()+l.wo.size()
       + l.wg.size()+l.wu.size()+l.wd.size()
       + l.rms_att.size()+l.rms_ffn.size();
  return n;
}

// ===========================================================================
//  Impl: every device buffer the forward pass touches.
// ===========================================================================
struct Model::Impl {
  // fp32 full-sequence buffers used by the reference / alignment path.
  // They exist to answer one question: if the model is run through the SAME
  // kernels the trainer uses, do its logits match the trainer's? If not, the
  // fp16 inference kernels are evaluating a different function than the one
  // the optimiser differentiated.
  GpuBuf f_x, f_xn, f_q, f_k, f_v, f_o, f_att, f_probs, f_rrms;
  GpuBuf f_gate, f_up, f_hmid, f_sig, f_tmp, f_logits;

  GpuBuf tok, x, xn, q, k, v, o, att;
  GpuBuf g, u, ff, logits;
  GpuBuf kcache, vcache;         // [n_layer][ctx][nkv*head_dim]
  GpuBuf tern_P, tern_M, tern_S; // one arena for all ternary layers
  GpuBuf emb, rmsf, lmh;
  GpuBuf scratch;
  std::vector<std::pair<size_t,size_t>> tern_off;   // [layer][7] -> (P,M,S offsets)
  cudaStream_t st = nullptr;
  int T = 0;
  bool ready = false;
  bool l2_engaged = false;
  L2Window l2;

  void teardown() {
    if (l2_engaged && st) { l2.release(st); l2_engaged = false; }
    if (st) { cudaStreamDestroy(st); st = nullptr; }
    ready = false;
  }
  ~Impl() { teardown(); }
};

void fill_rand_f16(std::vector<uint16_t>& v, std::mt19937& rng, float sigma) {
  std::normal_distribution<float> nd(0.f, sigma);
  for (auto& x : v) x = f32_to_f16(nd(rng));
}

Model::Model() : impl(new Impl) {}
Model::~Model() = default;

void Model::init(const Config& c) {
  cfg = c;
  layers.assign(c.n_layer, Layer());
  for (auto& L : layers) {
    L.wq.alloc(c.d_model, c.d_model);  L.wk.alloc(c.d_model, c.d_model);
    L.wv.alloc(c.d_model, c.d_model);  L.wo.alloc(c.d_model, c.d_model);
    L.w_gate.alloc(c.d_ff, c.d_model);
    L.w_up.alloc(c.d_ff, c.d_model);
    L.w_down.alloc(c.d_model, c.d_ff);
    L.rms_att.assign(c.d_model, f32_to_f16(1.0f));
    L.rms_ffn.assign(c.d_model, f32_to_f16(1.0f));
  }
  token_emb.assign((size_t)c.vocab * c.d_model, 0);
  rms_final.assign(c.d_model, f32_to_f16(1.0f));
  lm_head.assign((size_t)c.vocab * c.d_model, 0);
}

// ---------------------------------------------------------------------------
//  Ternary quantisation: threshold at a fraction of the per-row absolute mean.
//  Values above +thr -> +1, below -thr -> -1, otherwise 0.
//  The per-row scale is the mean absolute value of the surviving weights so
//  that the dequantised magnitudes stay in the original range.
//  (This is the standard TWN rule; we apply it per output row so that each
//  channel keeps its own dynamic range, at the cost of one fp16 per row.)
// ---------------------------------------------------------------------------
static void ternarize_row(const float* w, int K, uint32_t* P, uint32_t* M, float* out_scale) {
  double abssum = 0;
  for (int k = 0; k < K; ++k) abssum += std::fabs(w[k]);
  float thr = (float)(0.7 * abssum / K);
  double keep = 0; int nkeep = 0;
  for (int k = 0; k < K; ++k) {
    float v = w[k];
    uint32_t bit = 1u << (k & 31);
    size_t wi = (size_t)(k >> 5);
    if (v > thr)       { P[wi] |= bit; keep += std::fabs(v); nkeep++; }
    else if (v < -thr) { M[wi] |= bit; keep += std::fabs(v); nkeep++; }
  }
  *out_scale = nkeep ? (float)(keep / nkeep) : 1e-3f;
}

static void ternarize(const std::vector<float>& w, int N, int K, TernaryMatrix& out) {
  out.alloc(N, K);
  std::vector<float> sc(N);
  for (int n = 0; n < N; ++n)
    ternarize_row(&w[(size_t)n * K], K, &out.P[(size_t)n * (K / 32)], &out.M[(size_t)n * (K / 32)], &sc[n]);
  out.scale = sc.empty() ? 1.f : sc[0];
}

static void ternarize_lin(const std::vector<float>& w, TernLinear& L) {
  ternarize(w, L.out, L.in, L.W);
  // store per-row scale
  std::vector<float> sc(L.out);
  for (int n = 0; n < L.out; ++n) {
    double abssum = 0; int nz = 0;
    for (int k = 0; k < L.in; ++k) { double a = std::fabs(w[(size_t)n*L.in+k]); abssum += a; if (a>0) nz++; }
    float thr = (float)(0.7 * abssum / L.in);
    double keep = 0; int nk = 0;
    for (int k = 0; k < L.in; ++k) { float v = w[(size_t)n*L.in+k]; if (std::fabs(v) > thr) { keep += std::fabs(v); nk++; } }
    sc[n] = nk ? (float)(keep/nk) : 1e-3f;
  }
  for (int n = 0; n < L.out; ++n) L.scale[n] = f32_to_f16(sc[n]);
}

void Model::quantise_from(const MasterWeights& mw) {
  for (int i = 0; i < cfg.n_layer; ++i) {
    const auto& S = mw.layers[i];
    auto& L = layers[i];
    ternarize_lin(S.wq, L.wq); ternarize_lin(S.wk, L.wk);
    ternarize_lin(S.wv, L.wv); ternarize_lin(S.wo, L.wo);
    ternarize_lin(S.wg, L.w_gate); ternarize_lin(S.wu, L.w_up);
    ternarize_lin(S.wd, L.w_down);
    for (int d = 0; d < cfg.d_model; ++d) {
      L.rms_att[d] = f32_to_f16(S.rms_att[d]);
      L.rms_ffn[d] = f32_to_f16(S.rms_ffn[d]);
    }
  }
  for (size_t i = 0; i < token_emb.size(); ++i) token_emb[i] = f32_to_f16(mw.token_emb[i]);
  for (int d = 0; d < cfg.d_model; ++d) rms_final[d] = f32_to_f16(mw.rms_final[d]);
  // lm_head is tied to the (transposed) embedding when sizes match, else copied
  if (mw.lm_head.size() == lm_head.size()) {
    for (size_t i = 0; i < lm_head.size(); ++i) lm_head[i] = f32_to_f16(mw.lm_head[i]);
  }
}

// test hook: expose the row-wise ternariser
void ternarize_for_test(const std::vector<float>& w, int N, int K, TernaryMatrix& out) {
  ternarize(w, N, K, out);
}

size_t Model::ternary_weight_bytes() const {
  size_t b = 0;
  for (auto& L : layers) b += L.wq.W.weight_bytes() + L.wk.W.weight_bytes()
      + L.wv.W.weight_bytes() + L.wo.W.weight_bytes()
      + L.w_gate.W.weight_bytes() + L.w_up.W.weight_bytes() + L.w_down.W.weight_bytes();
  return b;
}
size_t Model::total_bytes() const {
  return ternary_weight_bytes()
    + token_emb.size()*2 + lm_head.size()*2 + rms_final.size()*2;
}
bool Model::fits_in_l2(size_t l2) const { return ternary_weight_bytes() <= l2; }

// ---------------------------------------------------------------------------
//  Device residency
// ---------------------------------------------------------------------------
void Model::to_device() {
  auto& I = *impl;
  if (!I.st) CUDA_CHECK(cudaStreamCreate(&I.st));
  const Config& c = cfg;
  const int d = c.d_model, ff = c.d_ff;
  const int nkv = c.n_head;   // MHA for now (GQA ready in the kernel)

  I.tok.alloc(4);
  I.x.alloc(d * 2); I.xn.alloc(d * 2);
  I.q.alloc(d * 2); I.k.alloc(d * 2); I.v.alloc(d * 2); I.o.alloc(d * 2); I.att.alloc(d * 2);
  I.g.alloc(ff * 2); I.u.alloc(ff * 2); I.ff.alloc(ff * 2);
  I.logits.alloc(c.vocab * 4);
  I.kcache.alloc((size_t)c.n_layer * c.ctx * nkv * c.head_dim() * 2);
  I.vcache.alloc((size_t)c.n_layer * c.ctx * nkv * c.head_dim() * 2);

  // ---- one contiguous arena for all ternary planes ----
  size_t per_layer_P = 0, per_layer_S = 0;
  for (auto& L : layers) {
    per_layer_P += L.wq.W.weight_bytes()/2 + L.wk.W.weight_bytes()/2 + L.wv.W.weight_bytes()/2
                 + L.wo.W.weight_bytes()/2 + L.w_gate.W.weight_bytes()/2
                 + L.w_up.W.weight_bytes()/2 + L.w_down.W.weight_bytes()/2;
    per_layer_S += (L.wq.out + L.wk.out + L.wv.out + L.wo.out
                  + L.w_gate.out + L.w_up.out + L.w_down.out) * 2;
  }
  size_t total_P = per_layer_P * c.n_layer;
  I.tern_P.alloc(total_P ? total_P : 4);
  I.tern_M.alloc(total_P ? total_P : 4);
  I.tern_S.alloc(per_layer_S * c.n_layer);

  // upload planes into the arena, recording offsets
  size_t offP = 0, offS = 0;
  std::vector<std::vector<std::pair<size_t,size_t>>> offs(c.n_layer);
  std::vector<uint8_t> hostP, hostM;
  std::vector<uint16_t> hostS;
  for (int i = 0; i < c.n_layer; ++i) {
    auto& L = layers[i];
    TernLinear* ls[7] = {&L.wq,&L.wk,&L.wv,&L.wo,&L.w_gate,&L.w_up,&L.w_down};
    for (int j = 0; j < 7; ++j) {
      size_t pb = ls[j]->W.weight_bytes()/2;
      offs[i].push_back({offP, offS});
      hostP.insert(hostP.end(), (uint8_t*)ls[j]->W.P.data(), (uint8_t*)ls[j]->W.P.data() + pb);
      hostM.insert(hostM.end(), (uint8_t*)ls[j]->W.M.data(), (uint8_t*)ls[j]->W.M.data() + pb);
      hostS.insert(hostS.end(), ls[j]->scale.begin(), ls[j]->scale.end());
      offP += pb; offS += ls[j]->scale.size()*2;
    }
  }
  I.tern_off.clear();
  for (auto& v : offs) for (auto& pr : v) I.tern_off.push_back(pr);

  if (!hostP.empty()) {
    CUDA_CHECK(cudaMemcpyAsync(I.tern_P.ptr, hostP.data(), hostP.size(), cudaMemcpyHostToDevice, I.st));
    CUDA_CHECK(cudaMemcpyAsync(I.tern_M.ptr, hostM.data(), hostM.size(), cudaMemcpyHostToDevice, I.st));
    CUDA_CHECK(cudaMemcpyAsync(I.tern_S.ptr, hostS.data(), hostS.size(), cudaMemcpyHostToDevice, I.st));
  }
  I.emb.alloc(token_emb.size()*2);
  I.lmh.alloc(lm_head.size()*2);
  I.rmsf.alloc(rms_final.size()*2);
  {
    const size_t LB = (size_t)c.ctx * 4;
    I.f_x.alloc(c.ctx*d*4);    I.f_xn.alloc(c.ctx*d*4);
    I.f_q.alloc(c.ctx*d*4);    I.f_k.alloc(c.ctx*d*4);
    I.f_v.alloc(c.ctx*d*4);    I.f_o.alloc(c.ctx*d*4);
    I.f_att.alloc(c.ctx*d*4);  I.f_tmp.alloc(c.ctx*d*4);
    I.f_probs.alloc(LB*(size_t)c.n_head*c.ctx);
    I.f_rrms.alloc(LB);
    I.f_gate.alloc(c.ctx*ff*4); I.f_up.alloc(c.ctx*ff*4);
    I.f_hmid.alloc(c.ctx*ff*4); I.f_sig.alloc(c.ctx*ff*4);
    I.f_logits.alloc(c.vocab*4);
  }
  CUDA_CHECK(cudaMemcpyAsync(I.emb.ptr, token_emb.data(), token_emb.size()*2, cudaMemcpyHostToDevice, I.st));
  CUDA_CHECK(cudaMemcpyAsync(I.lmh.ptr, lm_head.data(), lm_head.size()*2, cudaMemcpyHostToDevice, I.st));
  CUDA_CHECK(cudaMemcpyAsync(I.rmsf.ptr, rms_final.data(), rms_final.size()*2, cudaMemcpyHostToDevice, I.st));
  CUDA_CHECK(cudaStreamSynchronize(I.st));

  // rmsnorm tables per layer also need a home; pack them into tern_S tail
  // (kept simple: separate upload right after the scales)
  size_t rms_words = (size_t)c.n_layer * d * 2;
  I.scratch.alloc(rms_words * 2);
  std::vector<uint16_t> hrms;
  for (int i = 0; i < c.n_layer; ++i) {
    hrms.insert(hrms.end(), layers[i].rms_att.begin(), layers[i].rms_att.end());
    hrms.insert(hrms.end(), layers[i].rms_ffn.begin(), layers[i].rms_ffn.end());
  }
  CUDA_CHECK(cudaMemcpyAsync(I.scratch.ptr, hrms.data(), hrms.size()*2, cudaMemcpyHostToDevice, I.st));
  CUDA_CHECK(cudaStreamSynchronize(I.st));

  // ---- Innovation #2: pin the ternary arena into L2 ----
  size_t persist_max = query_l2_persist_max();
  size_t want = ternary_weight_bytes();
  if (want <= persist_max && want > 0) {
    size_t got = I.l2.engage(I.tern_P.ptr, want, I.st);
    if (got == want) I.l2_engaged = true;
    std::printf("[l2] pinned %.2f MiB of ternary planes into a %.2f MiB persistent window\n",
                got / 1048576.0, persist_max / 1048576.0);
  }
  I.ready = true;
  reset_kv();
}

void Model::reset_kv() {
  auto& I = *impl;
  I.T = 0;
  if (I.kcache.ptr) CUDA_CHECK(cudaMemsetAsync(I.kcache.ptr, 0, I.kcache.bytes, I.st));
  if (I.vcache.ptr) CUDA_CHECK(cudaMemsetAsync(I.vcache.ptr, 0, I.vcache.bytes, I.st));
  CUDA_CHECK(cudaStreamSynchronize(I.st));
}

// ---------------------------------------------------------------------------
//  Forward pass. One token per call, KV cache carries the context.
// ---------------------------------------------------------------------------
const std::vector<float>& Model::forward(int token, int pos, bool use_cache) {
  auto& I = *impl;
  const Config& c = cfg;
  const int d = c.d_model, ff = c.d_ff, hd = c.head_dim(), nh = c.n_head;
  cudaStream_t s = I.st;
  static std::vector<float> logits;

  int tok = token;
  CUDA_CHECK(cudaMemcpyAsync(I.tok.ptr, &tok, 4, cudaMemcpyHostToDevice, s));
  op_embed((const __half*)I.emb.ptr, (const int*)I.tok.ptr, (__half*)I.x.ptr, 1, d, s);

  if (!use_cache) I.T = 0;
  int T = use_cache ? (I.T + 1) : 1;

  const int stride = nh * hd;
  auto* kc = (__half*)I.kcache.ptr;
  auto* vc = (__half*)I.vcache.ptr;

  for (int l = 0; l < c.n_layer; ++l) {
    // ---- attention block ----
    op_rmsnorm((const __half*)I.x.ptr, (const __half*)((uint16_t*)I.scratch.ptr + (size_t)l*2*d),
               (__half*)I.xn.ptr, d, c.rms_eps, s);

    auto off = [&](int j) {
      return I.tern_off[(size_t)l*7 + j];
    };
    auto P = [&](int j) { return (const uint32_t*)((uint8_t*)I.tern_P.ptr + off(j).first); };
    auto M = [&](int j) { return (const uint32_t*)((uint8_t*)I.tern_M.ptr + off(j).first); };
    auto S = [&](int j) { return (const __half*)((uint8_t*)I.tern_S.ptr + off(j).second); };

    launch_gemv_t3_h(P(0), M(0), (const __half*)I.xn.ptr, S(0), (__half*)I.q.ptr, d, d, s);
    launch_gemv_t3_h(P(1), M(1), (const __half*)I.xn.ptr, S(1), (__half*)I.k.ptr, d, d, s);
    launch_gemv_t3_h(P(2), M(2), (const __half*)I.xn.ptr, S(2), (__half*)I.v.ptr, d, d, s);

    op_rope((__half*)I.q.ptr, (__half*)I.k.ptr, nh, hd, pos, c.rope_theta, s);

    // append k,v into the cache at slot T-1
    {
      __half* kdst = kc + ((size_t)l * c.ctx + (T - 1)) * stride;
      __half* vdst = vc + ((size_t)l * c.ctx + (T - 1)) * stride;
      CUDA_CHECK(cudaMemcpyAsync(kdst, I.k.ptr, stride * 2, cudaMemcpyDeviceToDevice, s));
      CUDA_CHECK(cudaMemcpyAsync(vdst, I.v.ptr, stride * 2, cudaMemcpyDeviceToDevice, s));
    }
    int smem = 4 * T * 4;   // NW=4 warps, T scores each
    op_attn_decode((const __half*)I.q.ptr, kc + (size_t)l * c.ctx * stride,
                   vc + (size_t)l * c.ctx * stride, (__half*)I.o.ptr,
                   T, nh, hd, nh, 1.0f / sqrtf((float)hd), smem, s);
    launch_gemv_t3_h(P(3), M(3), (const __half*)I.o.ptr, S(3), (__half*)I.att.ptr, d, d, s);

    op_add_inplace((__half*)I.x.ptr, (const __half*)I.att.ptr, d, s);

    // ---- ffn block ----
    op_rmsnorm((const __half*)I.x.ptr,
               (const __half*)((uint16_t*)I.scratch.ptr + ((size_t)l*2+1)*d),
               (__half*)I.xn.ptr, d, c.rms_eps, s);
    launch_gemv_t3_h(P(4), M(4), (const __half*)I.xn.ptr, S(4), (__half*)I.g.ptr, ff, d, s);
    launch_gemv_t3_h(P(5), M(5), (const __half*)I.xn.ptr, S(5), (__half*)I.u.ptr, ff, d, s);
    op_silu_mul((const __half*)I.g.ptr, (const __half*)I.u.ptr, (__half*)I.ff.ptr, ff, s);
    launch_gemv_t3_h(P(6), M(6), (const __half*)I.ff.ptr, S(6), (__half*)I.att.ptr, d, ff, s);
    op_add_inplace((__half*)I.x.ptr, (const __half*)I.att.ptr, d, s);
  }

  op_rmsnorm((const __half*)I.x.ptr, (const __half*)I.rmsf.ptr, (__half*)I.xn.ptr, d, c.rms_eps, s);
  op_lm_head((const __half*)I.lmh.ptr, (const __half*)I.xn.ptr, (float*)I.logits.ptr, c.vocab, d, s);

  logits.resize(c.vocab);
  CUDA_CHECK(cudaMemcpyAsync(logits.data(), I.logits.ptr, c.vocab*4, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  I.T = T;
  last_logits_copy = logits;   // sample() reads this
  return logits;
}

// ---------------------------------------------------------------------------
//  Sampling: temperature + top-k + top-p, all on the host (vocab is small).
// ---------------------------------------------------------------------------
int Model::sample(float temp, int top_k, float top_p, uint64_t seed) {
  static std::mt19937_64 rng(seed ? seed : 20261007ull);
  return sample_from(last_logits_copy, temp, top_k, top_p, rng);
}

int Model::sample_from(const std::vector<float>& lg, float temp, int top_k,
                       float top_p, std::mt19937_64& rng) {
  if (lg.empty()) return 0;
  if (temp <= 0.f) {   // greedy
    return (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
  }
  std::vector<int> idx(lg.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
  std::partial_sort(idx.begin(), idx.begin() + std::min<size_t>(idx.size(), std::max(top_k, 1)),
                    idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
  int k = std::min<int>(top_k > 0 ? top_k : (int)lg.size(), (int)lg.size());

  std::vector<double> p(k);
  double mx = lg[idx[0]];
  for (int i = 0; i < k; ++i) p[i] = std::exp((lg[idx[i]] - mx) / temp);
  double sum = 0; for (double v : p) sum += v;
  for (double& v : p) v /= sum;

  if (top_p < 1.0f) {
    double c = 0; int cut = k;
    for (int i = 0; i < k; ++i) { c += p[i]; if (c >= top_p) { cut = i + 1; break; } }
    p.resize(cut); sum = 0; for (double v : p) sum += v;
    for (double& v : p) v /= sum;
    k = cut;
  }
  std::discrete_distribution<int> dist(p.begin(), p.end());
  return idx[dist(rng)];
}

// ===========================================================================
//  Reference forward: identical kernels and precision to the trainer.
//
//  WHY THIS EXISTS
//  ---------------
//  The model has TWO forward implementations. model.cpp uses fp16 activations
//  with per-token KV-cache attention; ftrain_main.cpp uses fp32 activations
//  with full-sequence causal attention. A probe showed the two disagree by
//  556% with zero top-5 overlap, meaning training optimised one function and
//  inference evaluated another. This method runs the trainer's exact kernels
//  inside the inference model so the two can be compared directly.
// ===========================================================================
void Model::forward_fw(const std::vector<int>& ids, std::vector<float>& out_logits) {
  auto& I = *impl;
  const Config& c = cfg;
  const int d = c.d_model, ff = c.d_ff, hd = c.head_dim(), nh = c.n_head;
  const int L = (int)ids.size();
  cudaStream_t s = I.st;

  std::vector<int> host_ids = ids;
  CUDA_CHECK(cudaMemcpyAsync(I.tok.ptr, host_ids.data(), L*4, cudaMemcpyHostToDevice, s));

  // fp32 embedding gather
  for (int l = 0; l < c.n_layer; ++l) {}

  // embedding in fp32: convert the fp16 table on the fly via a small kernel
  extern void embed_fwd32(const float*, const int*, float*, int, int, cudaStream_t);
  // the fp16 table must be widened once; reuse f_x as staging is not safe, so
  // do the widening into f_xn then gather into f_x
  extern void launch_f16_to_f32(const __half*, float*, int, cudaStream_t);
  {
    // widen the whole table once into a scratch region (vocab*d floats)
    static GpuBuf wide;
    wide.alloc((size_t)c.vocab*d*4);
    launch_f16_to_f32((const __half*)I.emb.ptr, (float*)wide.ptr, c.vocab*d, s);
    embed_fwd32((const float*)wide.ptr, (const int*)I.tok.ptr, (float*)I.f_x.ptr, L, d, s);
  }
  extern void copy_f32(float*, const float*, int, cudaStream_t);
  copy_f32((float*)I.f_tmp.ptr, (const float*)I.f_x.ptr, L*d, s);

  for (int l = 0; l < c.n_layer; ++l) {
    auto off = [&](int j) { return I.tern_off[(size_t)l*7 + j]; };
    auto P = [&](int j) { return (const uint32_t*)((uint8_t*)I.tern_P.ptr + off(j).first); };
    auto M = [&](int j) { return (const uint32_t*)((uint8_t*)I.tern_M.ptr + off(j).first); };
    auto S = [&](int j) { return (const float*)((uint8_t*)I.tern_S.ptr + off(j).second); };

    const uint16_t* rms_att = (const uint16_t*)I.scratch.ptr + (size_t)l*2*d;
    const uint16_t* rms_ffn = rms_att + d;
    static GpuBuf r32; r32.alloc(d*4);
    extern void launch_f16_to_f32(const __half*, float*, int, cudaStream_t);
    launch_f16_to_f32((const __half*)rms_att, (float*)r32.ptr, d, s);

    extern void fw_rmsnorm(const float*, const float*, float*, float*, int, int, float, cudaStream_t);
    fw_rmsnorm((const float*)I.f_tmp.ptr, (const float*)r32.ptr,
               (float*)I.f_xn.ptr, (float*)I.f_rrms.ptr, L, d, c.rms_eps, s);

    extern void launch_tgemm_fwd(const float*, const uint32_t*, const uint32_t*, const float*,
                                 float*, int, int, int, cudaStream_t);
    launch_tgemm_fwd((const float*)I.f_xn.ptr, P(0), M(0), S(0), (float*)I.f_q.ptr, L, d, d, s);
    launch_tgemm_fwd((const float*)I.f_xn.ptr, P(1), M(1), S(1), (float*)I.f_k.ptr, L, d, d, s);
    launch_tgemm_fwd((const float*)I.f_xn.ptr, P(2), M(2), S(2), (float*)I.f_v.ptr, L, d, d, s);

    extern void fw_rope(float*, float*, int, int, int, float, cudaStream_t);
    fw_rope((float*)I.f_q.ptr, (float*)I.f_k.ptr, L, nh, hd, c.rope_theta, s);

    extern void fw_attn(const float*, const float*, const float*, float*, float*,
                        int, int, int, float, cudaStream_t);
    fw_attn((const float*)I.f_q.ptr, (const float*)I.f_k.ptr, (const float*)I.f_v.ptr,
            (float*)I.f_o.ptr, (float*)I.f_probs.ptr, L, nh, hd,
            1.0f / sqrtf((float)hd), s);

    launch_tgemm_fwd((const float*)I.f_o.ptr, P(3), M(3), S(3), (float*)I.f_tmp.ptr, L, d, d, s);
    extern void add_inplace32(float*, const float*, int, cudaStream_t);
    extern void copy_f32(float*, const float*, int, cudaStream_t);
    // f_tmp currently holds the attention output; f_x is the running residual
    add_inplace32((float*)I.f_x.ptr, (const float*)I.f_tmp.ptr, L*d, s);

    launch_f16_to_f32((const __half*)rms_ffn, (float*)r32.ptr, d, s);
    fw_rmsnorm((const float*)I.f_x.ptr, (const float*)r32.ptr,
               (float*)I.f_xn.ptr, (float*)I.f_rrms.ptr, L, d, c.rms_eps, s);

    launch_tgemm_fwd((const float*)I.f_xn.ptr, P(4), M(4), S(4), (float*)I.f_gate.ptr, L, ff, d, s);
    launch_tgemm_fwd((const float*)I.f_xn.ptr, P(5), M(5), S(5), (float*)I.f_up.ptr, L, ff, d, s);
    extern void fw_silu(const float*, const float*, float*, float*, int, cudaStream_t);
    fw_silu((const float*)I.f_gate.ptr, (const float*)I.f_up.ptr,
            (float*)I.f_hmid.ptr, (float*)I.f_sig.ptr, L*ff, s);
    launch_tgemm_fwd((const float*)I.f_hmid.ptr, P(6), M(6), S(6), (float*)I.f_tmp.ptr, L, d, ff, s);
    add_inplace32((float*)I.f_x.ptr, (const float*)I.f_tmp.ptr, L*d, s);
  }

  // final norm in fp32 with an fp32 copy of rms_final
  static GpuBuf rf32; rf32.alloc(d*4);
  {
    extern void launch_f16_to_f32(const __half*, float*, int, cudaStream_t);
    launch_f16_to_f32((const __half*)I.rmsf.ptr, (float*)rf32.ptr, d, s);
  }
  extern void fw_rmsnorm(const float*, const float*, float*, float*, int, int, float, cudaStream_t);
  fw_rmsnorm((const float*)I.f_x.ptr, (const float*)rf32.ptr,
             (float*)I.f_xn.ptr, (float*)I.f_rrms.ptr, L, d, c.rms_eps, s);

  const float* lastp = (const float*)I.f_xn.ptr + (size_t)(L-1)*d;
  extern void launch_dgemm_fwd(const float*, const float*, float*, int, int, int, cudaStream_t);
  launch_dgemm_fwd(lastp, (const float*)I.lmh.ptr == nullptr ? nullptr : nullptr, nullptr, 0, 0, 0, s);

  // lm_head in fp32
  static GpuBuf lh32; lh32.alloc((size_t)c.vocab*d*4);
  {
    extern void launch_f16_to_f32(const __half*, float*, int, cudaStream_t);
    launch_f16_to_f32((const __half*)I.lmh.ptr, (float*)lh32.ptr, c.vocab*d, s);
  }
  launch_dgemm_fwd(lastp, (const float*)lh32.ptr, (float*)I.f_logits.ptr, 1, c.vocab, d, s);

  out_logits.resize(c.vocab);
  CUDA_CHECK(cudaMemcpyAsync(out_logits.data(), I.f_logits.ptr, c.vocab*4,
                             cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
}

} // namespace canticle

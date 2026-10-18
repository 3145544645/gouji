#pragma once
#include "canticle/common.h"
namespace canticle {
void op_embed(const __half* tab, const int* tok, __half* out, int n, int d, cudaStream_t s);
void op_rmsnorm(const __half* x, const __half* w, __half* out, int d, float eps, cudaStream_t s);
void op_rope(__half* q, __half* k, int n_head, int head_dim, int pos, float theta, cudaStream_t s);
void op_silu_mul(const __half* g, const __half* u, __half* out, int n, cudaStream_t s);
void op_attn_decode(const __half* q, const __half* kc, const __half* vc, __half* o,
                    int T, int n_head, int head_dim, int nkv, float scale, int smem, cudaStream_t s);
void op_lm_head(const __half* head, const __half* x, float* logits, int vocab, int d, cudaStream_t s);
void op_gemv_f16(const __half* W, const __half* x, __half* y, int N, int K, cudaStream_t s);
void op_add_inplace(__half* a, const __half* b, int n, cudaStream_t s);
// ---- training-side launchers ----
void launch_tern_gemm_dx(const uint32_t* dP, const uint32_t* dM, const float* dscale,
                         const float* dY, float* dX, int N, int K, cudaStream_t s);
void launch_scale_sens(const float* raw, const float* dY, float* dscale, int N, cudaStream_t s);
void launch_meanpool(const __half* emb, const int* ids, __half* h, int T, int d, cudaStream_t s);
void launch_embed_bwd(float* demb, const int* ids, const float* dh, int T, int d, cudaStream_t s);
void launch_tern_head(const uint32_t* dP, const uint32_t* dM, const __half* h,
                      float* y, float* raw, int V, int d, cudaStream_t s);
void launch_ternarize_rows(const float* W, uint32_t* P, uint32_t* M, float* scale,
                           int N, int K, cudaStream_t s);
void launch_outer(float* dW, const float* g, const float* h, int N, int K, cudaStream_t s);
void launch_adam(float* W, const float* G, float* m, float* v, int n,
                 float lr, float b1, float b2, float eps, float wd,
                 float bc1, float bc2, cudaStream_t s);
void launch_f32_to_f16(const float* src, __half* dst, int n, cudaStream_t s);
void launch_f16_to_f32(const __half* src, float* dst, int n, cudaStream_t s);
void launch_scale_vec(const float* g, const float* scale, float* out, int n, cudaStream_t s);

// ---- full-model training launchers (fp32 activations, ternary weights) ----
void launch_tgemm_fwd(const float* X, const uint32_t* P, const uint32_t* M,
                      const float* scale, float* Y, int M_, int N, int K, cudaStream_t s);
void launch_tgemm_dx(const float* dY, const uint32_t* P, const uint32_t* M,
                     const float* scale, float* dX, int M_, int N, int K, cudaStream_t s);
void launch_tgemm_dW(const float* dY, const float* X, const float* scale,
                     float* dW, int M_, int N, int K, cudaStream_t s);
void launch_zero(float* p, int n, cudaStream_t s);
void launch_clip_grad(float* g, long n, float lim, cudaStream_t s);
void launch_nan_count(const float* g, long n, int* out, cudaStream_t s);
void launch_ce_rows(const float* logits, const int* target, float* dy, float* losses,
                    int L, int V, cudaStream_t s);
void launch_adamw2(float* W, const float* G, float* m, float* v, long n,
                   float lr, float b1, float b2, float eps, float wd,
                   float bc1, float bc2, cudaStream_t s);
void launch_mean_reduce(const float* a, float* out, int n, cudaStream_t s);
void launch_embed_bwd32(float* demb, const int* ids, const float* dh, int T, int d, cudaStream_t s);

void fw_rmsnorm(const float* x, const float* w, float* y, float* rrms,
                int L, int d, float eps, cudaStream_t s);
void bw_rmsnorm(const float* x, const float* w, const float* dy, const float* rrms,
                float* dx, float* dw, int L, int d, cudaStream_t s);
void fw_rope(float* q, float* k, int L, int H, int hd, float theta, cudaStream_t s);
void bw_rope(float* dq, float* dk, int L, int H, int hd, float theta, cudaStream_t s);
void fw_attn(const float* q, const float* k, const float* v, float* att, float* probs,
             int L, int H, int hd, float scale, cudaStream_t s);
void bw_attn(const float* q, const float* k, const float* v, const float* att,
             const float* probs, const float* datt, float* dprobs,
             float* dq, float* dk, float* dv, int L, int H, int hd, float scale,
             cudaStream_t s);
void fw_silu(const float* g, const float* u, float* h, float* sig, int n, cudaStream_t s);
void bw_silu(const float* g, const float* u, const float* sig, const float* dh,
             float* dg, float* du, int n, cudaStream_t s);
void add_inplace32(float* dst, const float* src, int n, cudaStream_t s);
void cast_f32_to_f16(const float* src, __half* dst, long n, cudaStream_t s);
void launch_dgemm_fwd(const float* X, const float* W, float* Y, int M_, int N, int K, cudaStream_t s);
void launch_dgemm_dx(const float* dY, const float* W, float* dX, int M_, int N, int K, cudaStream_t s);
void launch_dgemm_dW(const float* dY, const float* X, float* dW, int M_, int N, int K, cudaStream_t s);
void embed_fwd32(const float* emb, const int* ids, float* x, int L, int d, cudaStream_t s);
void copy_f32(float* dst, const float* src, int n, cudaStream_t s);
}

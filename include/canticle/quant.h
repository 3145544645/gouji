#pragma once
// Ternary (bitplane) quantization helpers.
//   W in {-1,0,+1}  ->  P (positive plane), M (negative plane), 2 bits per weight.
//   Invariant:  w = P - M.
// This is the primitive the whole CANTICLE inference path is built on.
#include "canticle/common.h"

namespace canticle {

// A packed ternary matrix: N rows x K cols, K must be a multiple of 32.
struct TernaryMatrix {
  int N = 0, K = 0;
  std::vector<uint32_t> P;   // N * (K/32)
  std::vector<uint32_t> M;   // N * (K/32)
  float scale = 1.0f;        // dequant scale (symmetric)

  void alloc(int n, int k) {
    N = n; K = k;
    size_t w = (size_t)n * (k / 32);
    P.assign(w, 0); M.assign(w, 0);
  }
  void set(int n, int k, int v) {
    size_t wi = (size_t)n * (K / 32) + (k / 32);
    uint32_t bit = 1u << (k & 31);
    if (v > 0)      P[wi] |= bit;
    else if (v < 0) M[wi] |= bit;
  }
  int get(int n, int k) const {
    size_t wi = (size_t)n * (K / 32) + (k / 32);
    uint32_t bit = 1u << (k & 31);
    return (int)((P[wi] & bit) != 0) - (int)((M[wi] & bit) != 0);
  }
  size_t weight_bytes() const { return (size_t)N * (K / 32) * 4 * 2; }
  double  bits_per_weight() const {
    return 8.0 * (double)weight_bytes() / (double)((size_t)N * K);
  }
};

// 4-bit symmetric quantization (group-size = whole row here, for a clean A/B).
struct Int4Matrix {
  int N = 0, K = 0;
  std::vector<uint32_t> W;   // 8 nibbles per uint32
  float scale = 1.0f;

  void alloc(int n, int k) {
    N = n; K = k;
    W.assign((size_t)n * (k / 8), 0);
  }
  void set(int n, int k, int q) {          // q in [-8, 7]
    uint32_t u = (uint32_t)(q + 8) & 0xF;
    size_t wi = (size_t)n * (K / 8) + (k / 8);
    W[wi] |= (u << (4 * (k & 7)));
  }
  int get(int n, int k) const {
    size_t wi = (size_t)n * (K / 8) + (k / 8);
    return (int)((W[wi] >> (4 * (k & 7))) & 0xF) - 8;
  }
  size_t weight_bytes() const { return (size_t)N * (K / 8) * 4; }
};

} // namespace canticle

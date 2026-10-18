// ===========================================================================
//  构屎 模型指纹
//
//  指纹不是序列号，也不是随机串。它由**模型状态本身**算出：
//  维度、意图数、工具数、阈值、判据数、变异数。
//
//  因此它满足两个必须的性质：
//    1. 同一状态 -> 同一指纹（可用来核对两次运行是否同一模型）
//    2. 任一状态量改变 -> 指纹改变（可用来发现「以为是同一个，其实不是」）
//
//  它**不是**安全机制。没有密钥，任何人都能算出同一指纹。
//  FNV-1a 的选择理由就是「短、快、无依赖」——不是为了抗碰撞。
// ===========================================================================
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>

namespace canticle {

namespace gouji {

// FNV-1a 32 位。逐字节混入状态量的文本表示，
// 这样「哪些量参与」在源码里是可见的——不像结构体哈希那样依赖内存布局。
struct Hash {
  uint32_t h = 2166136261u;
  void mix(const char* s) {
    while (*s) { h ^= (uint32_t)(unsigned char)*s++; h *= 16777619u; }
  }
  void mixf(float v) {
    char buf[32];
    // %.6f 而非 %g：0.15 与 0.150000 必须混出不同的字节串吗？不必——
    // 但固定小数位能让「看起来相同」的浮点数得到相同指纹，这正是要的。
    std::snprintf(buf, sizeof buf, "%.6f", (double)v);
    mix(buf);
  }
  void mixi(long long v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%lld", v);
    mix(buf);
  }
};

// 把 32 位拆成 4 组十六进制，形如 GOUJI-1A2B-3C4D-5E6F。
inline std::string format(uint32_t v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "GOUJI-%04X-%04X-%04X",
                (unsigned)(v >> 16) & 0xFFFFu,
                (unsigned)v & 0xFFFFu,
                (unsigned)((v >> 8) ^ (v << 5)) & 0xFFFFu);
  return std::string(buf);
}

} // namespace gouji
} // namespace canticle

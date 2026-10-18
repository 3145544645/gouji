// ===========================================================================
//  tokenizer.cpp -- 字节级 BPE（GPT-2 风格）+ 极简 JSON 取值工具
//
//  id 布局（train / save 之后保持稳定，load 也按此重建）：
//    0..5     保留特殊 token: <pad> <unk> <|user|> <|assistant|> <|tool|> <|/tool|>
//    6..261   256 个原始字节（id = 6 + byte），保证任意 UTF-8 / 二进制可编码
//    262..    合并 token（id = 262 + merge rank），add_special 追加在其后
//
//  关键不变量：
//    - 合并 token 由字节串拼接而来，因此永远不会等于任何特殊 token 串
//      （train 里对已存在的串直接禁用该 pair）；
//    - decode 只做拼接，于是 decode(encode(x)) == x 对任意输入成立；
//    - merge_rank 的 rank 唯一标识一个 pair，encode 靠它做“最小 rank 优先”贪心；
//    - 组成某一 merge 的两个 token 一定比它先出现（rank 更小），
//      所以优先队列贪心 == 教科书式的最低位 pair 反复合并。
//  无第三方依赖，纯 C++17。
// ===========================================================================
#include "canticle/model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace canticle {

// ---------------------------------------------------------------------------
//  内部工具
// ---------------------------------------------------------------------------
namespace {

constexpr int kNumSpecials = 6;                       // id 0..5
constexpr int kByteStart   = kNumSpecials;             // 6
constexpr int kFirstMerge  = kByteStart + 256;         // 262

const char* const kSpecialNames[kNumSpecials] = {
    "<pad>", "<unk>", "<|user|>", "<|assistant|>", "<|tool|>", "<|/tool|>"};

// 语料扫描上限与训练时间上限：保证 train() 一定在可接受时间内结束。
constexpr size_t kMaxCorpusBytes = (size_t)2 << 20;    // 2 MiB
constexpr double kTrainBudgetMs  = 8000.0;             // 8 s

// --- save/load 的转义：把 \ 与不可见字节写成 \n \r \t \xHH ----------------
// 原始字节里会出现 0x09/0x0a（一个 token 完全可能含换行），不转义就会破坏
// “一行一条”的格式，所以这里做一次可逆转义。
std::string escape_bytes(const std::string& s) {
  static const char* hex = "0123456789abcdef";
  std::string o;
  o.reserve(s.size() + 8);
  for (size_t i = 0; i < s.size(); ++i) {
    unsigned char c = (unsigned char)s[i];
    if (c == '\\')      o += "\\\\";
    else if (c == '\n') o += "\\n";
    else if (c == '\r') o += "\\r";
    else if (c == '\t') o += "\\t";
    else if (c < 0x20 || c == 0x7f) {
      o += "\\x"; o += hex[c >> 4]; o += hex[c & 15];
    } else {
      o += (char)c;
    }
  }
  return o;
}

int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

std::string unescape_bytes(const std::string& s) {
  std::string o;
  o.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (c != '\\' || i + 1 >= s.size()) { o += c; continue; }
    char n = s[++i];
    switch (n) {
      case '\\': o += '\\'; break;
      case 'n':  o += '\n'; break;
      case 'r':  o += '\r'; break;
      case 't':  o += '\t'; break;
      case 'x': {
        if (i + 2 < s.size()) {
          int hi = hexval(s[i + 1]), lo = hexval(s[i + 2]);
          if (hi >= 0 && lo >= 0) { o += (char)((hi << 4) | lo); i += 2; break; }
        }
        o += 'x';
        break;
      }
      default: o += n; break;
    }
  }
  return o;
}

// --- 极简 JSON -------------------------------------------------------------
bool json_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

// 找 "key" 后面跟冒号的位置（避免误命中“值是同名串”的情况），out = 冒号后第一个下标
bool json_find_value(const std::string& obj, const std::string& key, size_t& out) {
  const std::string pat = "\"" + key + "\"";
  size_t pos = 0;
  while ((pos = obj.find(pat, pos)) != std::string::npos) {
    size_t j = pos + pat.size();
    while (j < obj.size() && json_ws(obj[j])) ++j;
    if (j < obj.size() && obj[j] == ':') { out = j + 1; return true; }
    ++pos;
  }
  return false;
}

void utf8_append(std::string& o, unsigned cp) {
  if (cp < 0x80) {
    o += (char)cp;
  } else if (cp < 0x800) {
    o += (char)(0xC0 | (cp >> 6));
    o += (char)(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    o += (char)(0xE0 | (cp >> 12));
    o += (char)(0x80 | ((cp >> 6) & 0x3F));
    o += (char)(0x80 | (cp & 0x3F));
  } else {
    o += (char)(0xF0 | (cp >> 18));
    o += (char)(0x80 | ((cp >> 12) & 0x3F));
    o += (char)(0x80 | ((cp >> 6) & 0x3F));
    o += (char)(0x80 | (cp & 0x3F));
  }
}

}  // namespace

// ===========================================================================
//  BPETokenizer
// ===========================================================================

// 登记一个特殊 token；vocab 为空时先把 0..5 的基础特殊 token 建好，
// 这样 add_special 早早调用也不会把 id 0..5 占掉。
void BPETokenizer::add_special(const std::string& s) {
  if (s.empty()) return;
  if (vocab.empty()) {
    for (int i = 0; i < kNumSpecials; ++i) {
      vocab.push_back(kSpecialNames[i]);
      enc[kSpecialNames[i]] = i;
    }
  }
  if (enc.find(s) != enc.end()) return;         // 已存在则保持原 id
  enc[s] = (int)vocab.size();
  vocab.push_back(s);
}

int BPETokenizer::id_of(const std::string& s) const {
  std::unordered_map<std::string, int>::const_iterator it = enc.find(s);
  return it == enc.end() ? -1 : it->second;     // -1 = 未登记
}

// ---------------------------------------------------------------------------
//  train: 经典的“数相邻 pair -> 合并最高频的一个 -> 给下一个 id”循环。
//  做了增量化（链表 + 频次表 + 惰性最大堆），所以即使语料有 2 MiB 也能在
//  秒级跑完；再加时间上限兜底。
// ---------------------------------------------------------------------------
void BPETokenizer::train(const std::string& corpus, int target_vocab) {
  vocab.clear();
  enc.clear();
  merges.clear();
  merge_rank.clear();

  for (int i = 0; i < kNumSpecials; ++i) {
    vocab.push_back(kSpecialNames[i]);
    enc[kSpecialNames[i]] = i;
  }
  for (int b = 0; b < 256; ++b) {               // 基础字母表 = 256 个原始字节
    std::string s(1, (char)(unsigned char)b);
    vocab.push_back(s);
    enc[s] = kByteStart + b;
  }
  if (target_vocab <= kFirstMerge) return;

  const size_t n = std::min(corpus.size(), kMaxCorpusBytes);
  if (n < 2) return;

  std::vector<int> tok(n), prv(n), nxt(n);
  for (size_t i = 0; i < n; ++i) {
    tok[i] = kByteStart + (unsigned char)corpus[i];
    prv[i] = (i == 0) ? -1 : (int)i - 1;
    nxt[i] = (i + 1 < n) ? (int)i + 1 : -1;
  }

  std::unordered_map<uint64_t, int> cnt;        // pair -> 当前出现次数
  cnt.reserve(n * 2);
  std::priority_queue<std::pair<int, uint64_t> > pq;   // 最大堆（次数, pair）

  // inc 时把新次数压进堆；dec 到 0 就删除。堆里过期项在取用时惰性丢弃。
  std::unordered_map<uint64_t, int>* pcnt = &cnt;
  std::priority_queue<std::pair<int, uint64_t> >* ppq = &pq;
  auto inc = [pcnt, ppq](uint64_t k) {
    int c = ++(*pcnt)[k];
    ppq->push(std::make_pair(c, k));
  };
  auto dec = [pcnt](uint64_t k) {
    std::unordered_map<uint64_t, int>::iterator it = pcnt->find(k);
    if (it == pcnt->end()) return;
    if (--it->second <= 0) pcnt->erase(it);
  };

  for (size_t i = 0; i + 1 < n; ++i) inc(key(tok[i], tok[i + 1]));

  std::unordered_set<uint64_t> banned;          // 拼出来会撞已有 token 的 pair
  const double t0 = now_ms();
  const int head = 0;

  while ((int)vocab.size() < target_vocab) {
    if (now_ms() - t0 > kTrainBudgetMs) break;

    uint64_t bk = 0;
    int bc = 0;
    bool found = false;
    while (!pq.empty()) {                       // 取当前最高频的有效 pair
      const std::pair<int, uint64_t>& top = pq.top();
      int want = top.first;
      uint64_t k = top.second;
      pq.pop();
      std::unordered_map<uint64_t, int>::iterator it = cnt.find(k);
      if (it != cnt.end() && it->second == want && banned.find(k) == banned.end()) {
        bk = k; bc = want; found = true; break;
      }
    }
    if (!found || bc < 2) break;                // 只剩孤立 pair，停止合并

    const int a = (int)(uint32_t)(bk >> 32);
    const int b = (int)(uint32_t)(bk & 0xffffffffu);
    const std::string merged = vocab[a] + vocab[b];
    if (enc.find(merged) != enc.end()) {        // 与特殊 token / 旧 token 相撞
      banned.insert(bk);
      continue;
    }

    const int nid = (int)vocab.size();
    vocab.push_back(merged);
    enc[merged] = nid;
    merges.push_back(std::make_pair(a, b));
    merge_rank[bk] = (int)merges.size() - 1;

    // 单遍合并所有 (a,b) 相邻位，同时增量维护频次
    for (int i = head; i != -1;) {
      const int j = nxt[i];
      if (j != -1 && tok[i] == a && tok[j] == b) {
        const int p  = prv[i];
        const int nn = nxt[j];
        if (p  != -1) dec(key(tok[p], tok[i]));
        dec(bk);
        if (nn != -1) dec(key(tok[j], tok[nn]));
        tok[i] = nid;
        nxt[i] = nn;
        if (nn != -1) prv[nn] = i;              // j 变为死节点，不再可达
        if (p  != -1) inc(key(tok[p], tok[i]));
        if (nn != -1) inc(key(tok[i], tok[nn]));
        // 不前进：i 的右邻已经换人，由循环头重新判断
      } else {
        i = j;
      }
    }
  }
}

// ---------------------------------------------------------------------------
//  encode: 反复合并“rank 最小”的相邻 pair，直到没有可合并的。
//  用链表 + 最小堆，单个长串也是 O(n log n) 级别，不会退化。
// ---------------------------------------------------------------------------
std::vector<int> BPETokenizer::encode(const std::string& text) const {
  std::vector<int> ids;
  if (text.empty()) return ids;
  ids.reserve(text.size());

  int unk = -1;
  {
    std::unordered_map<std::string, int>::const_iterator it = enc.find("<unk>");
    if (it != enc.end()) unk = it->second;
  }
  for (size_t i = 0; i < text.size(); ++i) {
    std::string b(1, text[i]);
    std::unordered_map<std::string, int>::const_iterator it = enc.find(b);
    if (it != enc.end())      ids.push_back(it->second);
    else if (unk >= 0)        ids.push_back(unk);   // 理论上不会走到这里
  }

  const int n = (int)ids.size();
  if (n < 2 || merge_rank.empty()) return ids;

  std::vector<int> prv(n), nxt(n, -1);
  for (int i = 0; i < n; ++i) prv[i] = i - 1;
  for (int i = 0; i + 1 < n; ++i) nxt[i] = i + 1;

  typedef std::pair<int, int> Item;             // (rank, 左节点)
  std::priority_queue<Item, std::vector<Item>, std::greater<Item> > pq;
  for (int i = 0; i + 1 < n; ++i) {
    std::unordered_map<uint64_t, int>::const_iterator it =
        merge_rank.find(key(ids[i], ids[i + 1]));
    if (it != merge_rank.end()) pq.push(Item(it->second, i));
  }

  std::unordered_map<uint64_t, int> newid_cache;  // pair -> 合并后的 id
  while (!pq.empty()) {
    const Item top = pq.top();
    pq.pop();
    const int r = top.first;
    const int i = top.second;
    const int j = nxt[i];
    if (j < 0) continue;                        // 已是尾节点
    const uint64_t pk = key(ids[i], ids[j]);
    std::unordered_map<uint64_t, int>::const_iterator mr = merge_rank.find(pk);
    if (mr == merge_rank.end() || mr->second != r) continue;   // 过期堆项

    int nid;
    std::unordered_map<uint64_t, int>::const_iterator ci = newid_cache.find(pk);
    if (ci != newid_cache.end()) {
      nid = ci->second;
    } else {
      std::unordered_map<std::string, int>::const_iterator e =
          enc.find(vocab[ids[i]] + vocab[ids[j]]);
      if (e == enc.end()) continue;             // 不该发生；保守跳过
      nid = e->second;
      newid_cache[pk] = nid;
    }

    const int p  = prv[i];
    const int nn = nxt[j];
    ids[i] = nid;
    nxt[i] = nn;
    if (nn >= 0) prv[nn] = i;
    if (p >= 0) {
      std::unordered_map<uint64_t, int>::const_iterator e =
          merge_rank.find(key(ids[p], ids[i]));
      if (e != merge_rank.end()) pq.push(Item(e->second, p));
    }
    if (nn >= 0) {
      std::unordered_map<uint64_t, int>::const_iterator e =
          merge_rank.find(key(ids[i], ids[nn]));
      if (e != merge_rank.end()) pq.push(Item(e->second, i));
    }
  }

  std::vector<int> out;
  out.reserve(n);
  for (int i = 0; i >= 0 && i < n;) {           // 节点 0 永远是存活表头
    out.push_back(ids[i]);
    i = nxt[i];
  }
  return out;
}

// decode 只拼接，特殊 token 也照拼 —— 这样它才是 encode 的忠实逆。
std::string BPETokenizer::decode(const std::vector<int>& ids) const {
  std::string out;
  size_t total = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    const int id = ids[i];
    if (id >= 0 && id < (int)vocab.size()) total += vocab[id].size();
  }
  out.reserve(total);
  for (size_t i = 0; i < ids.size(); ++i) {
    const int id = ids[i];
    if (id >= 0 && id < (int)vocab.size()) out += vocab[id];
  }
  return out;
}

// ---------------------------------------------------------------------------
//  save / load：纯文本，一行一条 <id>\t<转义后的原始字节>。
//  load 时按 id 还原 vocab，并从“分裂点”重建 merges（两个子 token 的 id 必然
//  小于自身 id），因此 save->load 可完整往返。
// ---------------------------------------------------------------------------
bool BPETokenizer::save(const std::string& path) const {
  std::ofstream f(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!f) return false;
  for (size_t i = 0; i < vocab.size(); ++i) {
    f << i << '\t' << escape_bytes(vocab[i]) << '\n';
  }
  f.flush();
  return f.good();
}

bool BPETokenizer::load(const std::string& path) {
  std::ifstream f(path.c_str(), std::ios::binary);
  if (!f) return false;

  std::vector<std::pair<int, std::string> > entries;
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line[line.size() - 1] == '\r') line.erase(line.size() - 1);
    if (line.empty()) continue;
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) continue;              // 跳过非法行
    const int id = std::atoi(line.substr(0, tab).c_str());
    if (id < 0) continue;
    entries.push_back(std::make_pair(id, unescape_bytes(line.substr(tab + 1))));
  }
  if (entries.empty()) return false;

  int maxid = 0;
  for (size_t i = 0; i < entries.size(); ++i) maxid = std::max(maxid, entries[i].first);

  vocab.assign((size_t)maxid + 1, std::string());
  std::vector<char> has((size_t)maxid + 1, 0);
  enc.clear();
  for (size_t i = 0; i < entries.size(); ++i) {
    const int id = entries[i].first;
    if (has[id]) continue;
    has[id] = 1;
    vocab[id] = entries[i].second;
  }
  for (int i = 0; i <= maxid; ++i) {
    if (has[i] && !vocab[i].empty()) enc[vocab[i]] = i;
  }
  // 缺失的保留特殊 token 补登记，保证 <pad>/<unk>/... 一定可用
  for (int i = 0; i < kNumSpecials; ++i) {
    if (i > maxid || !has[i]) add_special(kSpecialNames[i]);
  }

  merges.clear();
  merge_rank.clear();
  for (int id = kFirstMerge; id <= maxid; ++id) {
    if (!has[id]) continue;
    const std::string& s = vocab[id];
    for (size_t len = 1; len < s.size(); ++len) {        // 找 a+b == s 的分裂点
      std::unordered_map<std::string, int>::const_iterator il = enc.find(s.substr(0, len));
      if (il == enc.end()) continue;
      std::unordered_map<std::string, int>::const_iterator ir = enc.find(s.substr(len));
      if (ir == enc.end()) continue;
      if (il->second < kByteStart || ir->second < kByteStart) continue;  // 不用特殊 token
      if (il->second >= id || ir->second >= id) continue;                // 子 token 必须更早
      merges.push_back(std::make_pair(il->second, ir->second));
      merge_rank[key(il->second, ir->second)] = (int)merges.size() - 1;
      break;
    }
  }
  return true;
}

// ===========================================================================
//  极简 JSON 取值工具（不依赖任何 JSON 库）
// ===========================================================================

std::string json_get_string(const std::string& obj, const std::string& key) {
  size_t i = 0;
  if (!json_find_value(obj, key, i)) return std::string();
  while (i < obj.size() && json_ws(obj[i])) ++i;
  if (i >= obj.size() || obj[i] != '"') return std::string();   // 非字符串
  ++i;
  std::string o;
  while (i < obj.size()) {
    const char c = obj[i++];
    if (c == '"') break;                     // 字符串结束
    if (c != '\\') { o += c; continue; }
    if (i >= obj.size()) break;
    const char e = obj[i++];
    switch (e) {
      case '"':  o += '"';  break;
      case '\\': o += '\\'; break;
      case '/':  o += '/';  break;
      case 'b':  o += '\b'; break;
      case 'f':  o += '\f'; break;
      case 'n':  o += '\n'; break;
      case 'r':  o += '\r'; break;
      case 't':  o += '\t'; break;
      case 'u': {
        unsigned cp = 0;
        int got = 0;
        while (got < 4 && i < obj.size()) {
          const int h = hexval(obj[i]);
          if (h < 0) break;
          cp = cp * 16u + (unsigned)h;
          ++i; ++got;
        }
        if (got < 4) break;
        if (cp >= 0xD800u && cp <= 0xDBFFu && i + 5 < obj.size() &&
            obj[i] == '\\' && obj[i + 1] == 'u') {          // UTF-16 代理对
          unsigned lo = 0;
          int got2 = 0;
          size_t k = i + 2;
          while (got2 < 4 && k < obj.size()) {
            const int h = hexval(obj[k]);
            if (h < 0) break;
            lo = lo * 16u + (unsigned)h;
            ++k; ++got2;
          }
          if (got2 == 4 && lo >= 0xDC00u && lo <= 0xDFFFu) {
            cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
            i = k;
          }
        }
        utf8_append(o, cp);
        break;
      }
      default: o += e; break;
    }
  }
  return o;
}

double json_get_number(const std::string& obj, const std::string& key) {
  size_t i = 0;
  if (!json_find_value(obj, key, i)) return 0.0;
  while (i < obj.size() && json_ws(obj[i])) ++i;
  if (i >= obj.size()) return 0.0;
  const char c = obj[i];
  if (!(c == '-' || c == '+' || (c >= '0' && c <= '9'))) return 0.0;  // 非数字字面量
  const char* begin = obj.c_str() + i;
  char* end = nullptr;
  const double v = std::strtod(begin, &end);
  if (end == begin) return 0.0;
  return v;
}

}  // namespace canticle

// ===========================================================================
//  构屎 指纹：把模型状态压成一个可核对的短串
//
//  设计上的两个决定：
//
//  1. 参与指纹的量**逐条打印**，而不是只给一个最终哈希。
//     只给哈希的话，指纹变了也无法知道是哪一项变了——
//     那就成了一个「告诉你有问题但不告诉你在哪」的装置，
//     和本工程反复修的「装置失效」是同一类（判据 1）。
//
//  2. 只混入**状态量**，不混入时间、路径、内存地址。
//     否则同一模型在不同机器上得到不同指纹，指纹就失去了意义——
//     它要回答的是「这是不是同一个模型」，不是「这是不是同一次运行」。
// ===========================================================================
#include "canticle/fingerprint.h"
#include "canticle/am.h"
#include "canticle/model.h"
#include "canticle/learn.h"
#include "canticle/mutate.h"
#include <cstdio>
#include <fstream>
#include <sstream>

using namespace canticle;

namespace {

// main.cpp 里的两个下限定义在 .cpp 内，这里按源码抽取，
// 而不是再写一份字面量——判据 2。
bool scrape_float(const char* file, const char* key, float* out) {
  std::ifstream f(file, std::ios::binary);
  if (!f) return false;
  std::stringstream ss; ss << f.rdbuf();
  const std::string s = ss.str();
  const size_t k = s.find(key);
  if (k == std::string::npos) return false;
  const size_t eq = s.find('=', k);
  if (eq == std::string::npos) return false;
  *out = (float)std::atof(s.c_str() + eq + 1);
  return true;
}

// 判据条数：数 CRITERIA.md 里的二级标题。
int count_criteria() {
  std::ifstream f("CRITERIA.md", std::ios::binary);
  if (!f) return -1;
  std::stringstream ss; ss << f.rdbuf();
  const std::string s = ss.str();
  int n = 0;
  for (size_t p = s.find("## 判据"); p != std::string::npos;
       p = s.find("## 判据", p + 1)) n++;
  return n;
}

} // namespace

int am_fingerprint_run(int argc, char** argv) {
  const bool quiet = (argc > 1 && std::string(argv[1]) == "--quiet");

  gouji::Hash h;

  // ---- 表示层 ----
  const int dim = 2048;                       // AmMem::D
  const int intents = (int)Intent::Unsupported;
  const int tools = (int)tool_specs().size();
  const int buckets = kIntentBucketCount;

  // ---- 阈值 ----
  float floor_known = 0.f, floor_unknown = 0.f;
  const bool got_known   = scrape_float("src/am_main.cpp", "MIN_MATCH_STRUCTURED", &floor_known);
  const bool got_unknown = scrape_float("src/am_main.cpp", "MIN_MATCH_UNKNOWN", &floor_unknown);
  const float ceiling = SIM_CEILING;

  // ---- 工程状态 ----
  const int criteria = count_criteria();
  // Named differently from the function it calls: "mutations" as a variable shadows
  // the accessor and the compiler reports it as "term does not evaluate to a
  // function taking 0 arguments", which reads like a broken header rather than a
  // name collision.
  const auto& mut_table = mutations();
  const int mut_count = (int)mut_table.size();
  int suites = 0;
  {
    std::vector<std::string> seen;
    for (const Mutation& m : mut_table) {
      bool dup = false;
      for (const auto& s : seen) if (s == m.suite) { dup = true; break; }
      if (!dup) seen.push_back(m.suite);
    }
    suites = (int)seen.size();
  }

  // 混入的顺序固定，且每一项都带一个标签，
  // 这样「维度 2048 意图 6」与「维度 6 意图 2048」不会撞成同一串。
  h.mix("dim=");      h.mixi(dim);
  h.mix(";intents="); h.mixi(intents);
  h.mix(";tools=");   h.mixi(tools);
  h.mix(";buckets="); h.mixi(buckets);
  h.mix(";floorK=");  h.mixf(floor_known);
  h.mix(";floorU=");  h.mixf(floor_unknown);
  h.mix(";ceiling="); h.mixf(ceiling);
  h.mix(";crit=");    h.mixi(criteria);
  h.mix(";mut=");     h.mixi(mut_count);
  h.mix(";suites=");  h.mixi(suites);

  const std::string fp = gouji::format(h.h);

  if (quiet) { std::printf("%s\n", fp.c_str()); return 0; }

  std::printf("=== 构屎 GOUJI ===\n\n");
  std::printf("  指纹  %s\n\n", fp.c_str());
  std::printf("  参与指纹的状态量（改任一项，指纹必变）：\n");
  std::printf("    维度 D            %d\n", dim);
  std::printf("    意图数            %d\n", intents);
  std::printf("    工具数            %d\n", tools);
  std::printf("    意图桶数          %d\n", buckets);
  if (got_known)   std::printf("    下限(意图已知)    %.6f\n", (double)floor_known);
  else             std::printf("    下限(意图已知)    <未读到 src/am_main.cpp>\n");
  if (got_unknown) std::printf("    下限(意图未知)    %.6f\n", (double)floor_unknown);
  else             std::printf("    下限(意图未知)    <未读到 src/am_main.cpp>\n");
  std::printf("    冗余上限          %.6f\n", (double)ceiling);
  if (criteria >= 0) std::printf("    判据条数          %d\n", criteria);
  else               std::printf("    判据条数          <未读到 CRITERIA.md>\n");
  std::printf("    变异数            %d\n", mut_count);
  std::printf("    覆盖套件数        %d\n", suites);

  // 读不到输入时指纹仍在，但它是「残缺状态的指纹」。
  // 这一点必须说出来：否则一个缺了三个量的指纹会看起来和完整的一样权威。
  if (!got_known || !got_unknown || criteria < 0) {
    std::printf("  注意：有状态量未读到，此指纹不完整。\n");
    std::printf("        请在项目根目录运行，且先执行 collect/criteria 相关步骤。\n");
  }
  return 0;
}

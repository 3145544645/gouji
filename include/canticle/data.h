#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
namespace canticle {
struct Sample { std::string prompt; std::string reply; };
std::vector<Sample> synth_tool_data(int n, uint64_t seed);

// Per-tool sample counts from the most recent synth_tool_data call.
//
// Exposed because the generator's schedule is uneven by construction (see the
// comment in data.cpp): the newer tools appear every 7th, 11th and 13th iteration,
// so a modest n yields very few of them. A caller that samples "the first few of
// each tool" can collect nothing and still look like it measured something --
// which is exactly what defect 67 did, reporting a confident "unknown" baseline
// built from empty buckets.
//
// An empty map means no generation has run yet.
namespace detail {
std::unordered_map<std::string, int>& last_tool_coverage();
}
}

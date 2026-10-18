#pragma once
#include "canticle/model.h"
namespace canticle {
bool save_checkpoint(const std::string& path, const Model& m, const BPETokenizer* tk);
bool load_checkpoint(const std::string& path, Model& m, BPETokenizer* tk);
}

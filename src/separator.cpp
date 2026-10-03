// separator.cpp — architecture dispatch for load_separator(). See separator.h.
#include "separator.h"

#include "gguf_model.h"
#include "htdemucs.h"
#include "roformer.h"

namespace st {

std::string gguf_architecture(const std::string& path) {
    const GgufModel m = load_gguf_metadata(path.c_str());
    return m.string("general.architecture");
}

std::unique_ptr<Separator> load_separator(const std::string& path, const char* device, int cpu_threads) {
    const std::string arch = gguf_architecture(path);
    if (arch == "htdemucs") return std::make_unique<HTDemucs>(path, device, cpu_threads);
    if (arch == "mel_band_roformer" || arch == "bs_roformer")
        return std::make_unique<RoFormer>(path, device, cpu_threads);
    throw std::runtime_error(path + ": unsupported architecture '" + arch + "'");
}

} // namespace st

// UVR 5.1 CascadedNet: one metadata-driven network for compatible VR checkpoints.
#pragma once
#include "separator.h"
#include <memory>

namespace st {
class VR final : public Separator {
public:
    explicit VR(const std::string& path, const char* device = nullptr, int cpu_threads = 0);
    ~VR() override;
    const std::string& name() const override;
    const std::string& architecture() const override;
    const std::vector<std::string>& sources() const override;
    int samplerate() const override;
    int audio_channels() const override { return 2; }
    int segment_samples() const override;
    int n_models() const override { return 1; }
    const char* backend_name() const override;
    std::vector<float> separate(const float* mix, int len, const SeparateOptions& opt = {}) const override;
    // Normalized magnitude [channel][bins+1][window] -> cropped primary mask
    // [channel][bins+1][window-2*offset]. Independent network parity entry point.
    std::vector<float> predict_mask(const float* magnitude) const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
} // namespace st

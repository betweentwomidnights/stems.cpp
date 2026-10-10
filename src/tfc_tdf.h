// tfc_tdf.h — TFC-TDF U-Nets over a complex spectrogram: UVR's MDX-Net (ConvTDFNet, as
// exported to ONNX) and ZFTurbo's MDX23C (TFC_TDF_net v3). One GGUF architecture, "tfc_tdf";
// tfc_tdf.variant selects the block structure and the chunking. See docs/MDX.md.
#pragma once

#include "separator.h"

#include <memory>

namespace st {

class TFCTDF final : public Separator {
public:
    explicit TFCTDF(const std::string& path, const char* device = nullptr, int cpu_threads = 0);
    ~TFCTDF() override;

    const std::string& name() const override;
    const std::string& architecture() const override;
    const std::vector<std::string>& sources() const override;   // model stems, then the complement
    int samplerate() const override;
    int audio_channels() const override { return 2; }
    int segment_samples() const override;                       // hop * (dim_t - 1)
    int n_models() const override { return 1; }
    const char* backend_name() const override;

    std::vector<float> separate(const float* mix, int len, const SeparateOptions& opt = {}) const override;

    // One chunk, planar [2][segment_samples()], through STFT, network and iSTFT: the model's own
    // stems [S][2][segment_samples()], before any windowing or volume compensation. Parity entry.
    std::vector<float> forward(const float* chunk) const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace st

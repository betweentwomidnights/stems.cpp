// roformer.h — Band-Split / Mel-Band RoFormer source separation on ggml.
//
// One class covers both architectures (lucidrains' BS-RoFormer and Mel-Band RoFormer, as
// trained by ZFTurbo's Music-Source-Separation-Training and its descendants). They differ only
// in how STFT bins are grouped into bands: BS-RoFormer uses contiguous ranges, Mel-Band uses
// overlapping mel-spaced bands whose masks are averaged where they overlap. The converter
// (tools/convert_roformer.py) writes that grouping into the GGUF as plain index lists, so here
// both are the same code.
//
// The network runs as one ggml graph per chunk (8 s for the published checkpoints): band split
// -> depth x (time transformer, freq transformer) with rotary embeddings -> one mask-estimator
// MLP per band and stem. The STFT, mask application, iSTFT and the chunk/overlap/fade
// bookkeeping (demix_track) run on the host.
#pragma once

#include "separator.h"

#include <memory>
#include <string>
#include <vector>

namespace st {

class RoFormer final : public Separator {
public:
    explicit RoFormer(const std::string& gguf_path, const char* device = nullptr, int cpu_threads = 0);
    ~RoFormer() override;
    RoFormer(const RoFormer&) = delete;
    RoFormer& operator=(const RoFormer&) = delete;

    const std::string& name() const override;
    const std::string& architecture() const override;
    // The model's own stems, then the complement (mix minus the first stem) when the GGUF
    // declares one: Kim's vocal model gives {"vocals", "instrumental"}.
    const std::vector<std::string>& sources() const override;
    int samplerate() const override;
    int audio_channels() const override;
    int segment_samples() const override;   // the chunk size
    int n_models() const override { return 1; }
    const char* backend_name() const override;

    std::vector<float> separate(const float* mix, int len, const SeparateOptions& opt = {}) const override;

    // One raw model forward on exactly segment_samples() of audio: planar [C][N] -> [S][C][N]
    // for the model's own stems (no complement). For the parity tests.
    std::vector<float> forward(const float* chunk) const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace st

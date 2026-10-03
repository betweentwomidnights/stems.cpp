// htdemucs.h — Hybrid Transformer Demucs (v4) source separation on ggml.
//
// One class covers every published HTDemucs checkpoint: `htdemucs` (4 stems), `htdemucs_6s`
// (6 stems) and `htdemucs_ft` (a bag of four per-source fine-tunes). The network runs as one
// ggml graph per 7.8 s segment; the spectrogram, normalisation and the segment/overlap/shift
// bookkeeping of demucs.apply run on the host, reproduced exactly so outputs match
// `demucs --shifts 0` sample for sample (see docs/PARITY.md).
#pragma once

#include "separator.h"

#include <memory>
#include <string>
#include <vector>

namespace st {

class HTDemucs final : public Separator {
public:
    // device: nullptr/"" = best GPU (STEMS_DEVICE / STEMS_GPU env vars apply), "cpu" = CPU.
    explicit HTDemucs(const std::string& gguf_path, const char* device = nullptr, int cpu_threads = 0);
    ~HTDemucs() override;
    HTDemucs(const HTDemucs&) = delete;
    HTDemucs& operator=(const HTDemucs&) = delete;

    const std::string& name() const override;
    const std::string& architecture() const override;
    const std::vector<std::string>& sources() const override;
    int samplerate() const override;
    int audio_channels() const override;
    int segment_samples() const override;
    int n_models() const override;
    const char* backend_name() const override;

    // Full separation. mix is planar [audio_channels][len] at samplerate(); returns planar
    // [S][audio_channels][len]. Normalises by the mono mix exactly as `demucs` does.
    std::vector<float> separate(const float* mix, int len, const SeparateOptions& opt = {}) const override;

    // One raw model forward on exactly segment_samples() of already-normalised audio:
    // planar [C][N] -> [S][C][N]. For the parity tests. If dump_dir is non-empty, the
    // intermediate activations are written there as raw float32 (same memory layout as
    // the PyTorch arrays in tests/refs).
    std::vector<float> forward(int model_index, const float* seg, const std::string& dump_dir = "") const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace st

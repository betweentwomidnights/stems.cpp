// separator.h — the interface every stems.cpp model architecture implements.
//
// A GGUF names its architecture in `general.architecture`; load_separator() reads that and
// constructs the matching implementation, so libstems, stems-split and stems-server never name
// a concrete model class. Every implementation takes planar audio at its own samplerate() and
// audio_channels() and returns planar stems in sources() order; resampling and channel
// conversion stay with the caller (see audio.h).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace st {

struct SeparateOptions {
    // Random time shifts averaged for time equivariance (demucs --shifts). Each shift is a
    // full extra pass. 0 = one deterministic pass.
    int shifts = 0;
    // Overlap between consecutive segments, [0, 1). Negative = the model's own default
    // (0.25 for HTDemucs, the checkpoint's num_overlap for RoFormers).
    float overlap = -1.0f;
    // Seed for the shift offsets.
    uint32_t seed = 0;
    // Called with (done, total) segment passes.
    std::function<void(int, int)> progress;
    // Polled before every segment pass; returning true abandons the job with Cancelled.
    std::function<bool()> should_cancel;
};

struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("cancelled") {}
};

class Separator {
public:
    virtual ~Separator() = default;

    virtual const std::string& name() const = 0;           // checkpoint, e.g. "htdemucs_6s"
    virtual const std::string& architecture() const = 0;   // GGUF general.architecture
    virtual const std::vector<std::string>& sources() const = 0;
    virtual int samplerate() const = 0;
    virtual int audio_channels() const = 0;
    virtual int segment_samples() const = 0;
    virtual int n_models() const = 0;                      // > 1 for a bag of models
    virtual const char* backend_name() const = 0;

    // mix is planar [audio_channels()][len] at samplerate(); returns planar
    // [sources().size()][audio_channels()][len].
    virtual std::vector<float> separate(const float* mix, int len, const SeparateOptions& opt = {}) const = 0;
};

// Opens any stems.cpp GGUF. device: nullptr/"" = best GPU (STEMS_DEVICE / STEMS_GPU env vars
// apply), "cpu" = CPU. Throws std::runtime_error for an unknown architecture.
std::unique_ptr<Separator> load_separator(const std::string& gguf_path, const char* device = nullptr,
                                          int cpu_threads = 0);

// The architecture a GGUF declares, read from its metadata alone (no weights).
std::string gguf_architecture(const std::string& gguf_path);

} // namespace st

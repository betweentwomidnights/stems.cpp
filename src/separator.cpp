// separator.cpp — architecture dispatch for load_separator(). See separator.h.
#include "separator.h"

#include "gguf_model.h"
#include "htdemucs.h"
#include "roformer.h"
#include "vr.h"

#include <algorithm>

namespace st {

namespace {

// A mono network presented as stereo: each channel goes through the network on its own and
// the results are interleaved back into planar stereo. Callers convert to audio_channels()
// before separate(), and stereo through a mono model would otherwise keep only the left
// channel; dual mono keeps the stereo image and every published model's output stereo.
class DualMono final : public Separator {
public:
    explicit DualMono(std::unique_ptr<Separator> m) : m_(std::move(m)) {}

    const std::string& name() const override { return m_->name(); }
    const std::string& architecture() const override { return m_->architecture(); }
    const std::vector<std::string>& sources() const override { return m_->sources(); }
    int samplerate() const override { return m_->samplerate(); }
    int audio_channels() const override { return 2; }
    int segment_samples() const override { return m_->segment_samples(); }
    int n_models() const override { return m_->n_models(); }
    const char* backend_name() const override { return m_->backend_name(); }

    std::vector<float> separate(const float* mix, int len, const SeparateOptions& opt) const override {
        const size_t S = m_->sources().size(), n = (size_t)len;
        std::vector<float> out(S * 2 * n);
        for (int c = 0; c < 2; c++) {
            SeparateOptions o = opt;
            if (opt.progress)   // one run per channel: report them as one job twice as long
                o.progress = [&opt, c](int done, int total) { opt.progress(c * total + done, 2 * total); };
            const std::vector<float> y = m_->separate(mix + c * n, len, o);
            for (size_t s = 0; s < S; s++)
                std::copy(y.begin() + s * n, y.begin() + (s + 1) * n, out.begin() + (s * 2 + c) * n);
        }
        return out;
    }

private:
    std::unique_ptr<Separator> m_;
};

std::unique_ptr<Separator> stereo(std::unique_ptr<Separator> m) {
    if (m->audio_channels() == 1) return std::make_unique<DualMono>(std::move(m));
    return m;
}

} // namespace

std::string gguf_architecture(const std::string& path) {
    const GgufModel m = load_gguf_metadata(path.c_str());
    return m.string("general.architecture");
}

std::unique_ptr<Separator> load_separator(const std::string& path, const char* device, int cpu_threads) {
    const std::string arch = gguf_architecture(path);
    if (arch == "htdemucs") return std::make_unique<HTDemucs>(path, device, cpu_threads);
    if (arch == "vr_cascaded") return std::make_unique<VR>(path, device, cpu_threads);
    if (arch == "mel_band_roformer" || arch == "bs_roformer")
        return stereo(std::make_unique<RoFormer>(path, device, cpu_threads));
    throw std::runtime_error(path + ": unsupported architecture '" + arch + "'");
}

} // namespace st

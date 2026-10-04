// dsp_test: the host-side signal path every parity number rests on, with no model.
//   - the FFT inverts itself;
//   - torch_stft -> torch_istft is the identity (RoFormer's n_fft 2048, hop 441);
//   - htdemucs_spec -> htdemucs_ispec is the identity on band-limited audio away from the edges
//     (HTDemucs drops the Nyquist bin and two edge frames each side by design);
//   - WAV bytes round-trip, float32 exactly and 16-bit to within its quantization;
//   - the resampler reproduces a sine at the new rate and is a no-op at the same rate.
#include "audio.h"
#include "stft.h"
#include "wav.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what, double value, double limit) {
    printf("  %-58s %.3e (limit %.0e) %s\n", what, value, limit, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

double max_abs_diff(const float* a, const float* b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

// Sum of sines well below Nyquist, so nothing lives in the bin HTDemucs drops.
std::vector<float> band_limited(int n, int sr, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> freq(40.0, 0.4 * sr), phase(0.0, 2 * M_PI), amp(0.02, 0.2);
    std::vector<double> f(24), p(24), a(24);
    for (int k = 0; k < 24; k++) { f[k] = freq(rng); p[k] = phase(rng); a[k] = amp(rng); }
    std::vector<float> x(n);
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int k = 0; k < 24; k++) s += a[k] * std::sin(2 * M_PI * f[k] * i / sr + p[k]);
        x[i] = (float)s;
    }
    return x;
}

} // namespace

int main() {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

    printf("fft\n");
    {
        st::FFT fft(4096);
        std::vector<st::cplx> a(4096), orig(4096);
        for (auto& v : a) v = st::cplx(uni(rng), uni(rng));
        orig = a;
        fft.run(a, false);
        fft.run(a, true);
        double m = 0;
        for (int i = 0; i < 4096; i++) m = std::max(m, std::abs(a[i] / 4096.0 - orig[i]));
        check(m < 1e-12, "forward then inverse / n", m, 1e-12);
    }

    printf("torch_stft / torch_istft (n_fft 2048, hop 441)\n");
    for (bool normalized : {false, true}) {
        st::FFT fft(2048);
        const int hop = 441, len = hop * 200;
        std::vector<float> x(len);
        for (auto& v : x) v = uni(rng);
        int T = 0;
        const auto spec = st::torch_stft(fft, x.data(), len, hop, normalized, T);
        const auto y = st::torch_istft(fft, spec, T, hop, normalized);
        const double m = max_abs_diff(x.data(), y.data(), y.size());
        check(y.size() == (size_t)len && m < 1e-5,
              normalized ? "round trip, normalized" : "round trip", m, 1e-5);
    }

    printf("htdemucs_spec / htdemucs_ispec (n_fft 4096, hop 1024)\n");
    {
        st::FFT fft(4096);
        const int len = 1024 * 60;
        const auto x = band_limited(len, 44100, 7);
        int T = 0;
        const auto spec = st::htdemucs_spec(fft, x.data(), len, T);
        const auto y = st::htdemucs_ispec(fft, spec, T, len);
        // Near the ends it is not an identity, by design (PyTorch's _spec/_ispec behave the
        // same): _spec drops two frames at each end, and the frames straddling the reflect-pad
        // seam carry broadband energy into the Nyquist bin it drops. Both reach at most
        // 3.5 hops in; in the model the time branch covers that. Past nfft from either end it
        // is an identity to ~1e-9.
        const int edge = 4096;
        const double m = max_abs_diff(x.data() + edge, y.data() + edge, len - 2 * edge);
        check(T == len / 1024 && m < 1e-6, "round trip, band-limited, away from the edges", m, 1e-6);
    }

    printf("wav\n");
    {
        const int n = 4410, ch = 2;
        std::vector<float> x((size_t)n * ch);
        for (auto& v : x) v = 0.9f * uni(rng);
        int rn = 0, rc = 0, rsr = 0;
        const auto f32 = st::parse_wav_planar(st::wav_bytes(x.data(), n, ch, 44100, true), rn, rc, rsr);
        const double mf = max_abs_diff(x.data(), f32.data(), x.size());
        check(rn == n && rc == ch && rsr == 44100 && mf == 0.0, "float32 round trip", mf, 0.0);
        const auto i16 = st::parse_wav_planar(st::wav_bytes(x.data(), n, ch, 48000, false), rn, rc, rsr);
        const double mi = max_abs_diff(x.data(), i16.data(), x.size());
        // Written as lrint(x * 32767), read as / 32768 (the usual pair): half an LSB of rounding
        // plus |x| / 32768 of scale, so at most 1.5 / 32768 at full scale.
        check(rn == n && rc == ch && rsr == 48000 && mi <= 1.5 / 32768, "16-bit round trip", mi, 1.5 / 32768);
    }

    printf("resample\n");
    {
        int out_n = 0;
        const int n = 48000;
        std::vector<float> x(n);
        for (int i = 0; i < n; i++) x[i] = 0.5f * (float)std::sin(2 * M_PI * 1000.0 * i / 48000);
        const auto same = st::resample_planar(x, n, 1, 48000, 48000, out_n);
        check(out_n == n && max_abs_diff(x.data(), same.data(), n) == 0.0, "same rate is a no-op", 0.0, 0.0);
        const auto y = st::resample_planar(x, n, 1, 48000, 44100, out_n);
        // Compare away from the edges, where the windowed sinc runs out of input.
        double m = 0;
        for (int i = 1000; i < out_n - 1000; i++)
            m = std::max(m, std::fabs(y[i] - 0.5 * std::sin(2 * M_PI * 1000.0 * i / 44100)));
        check(out_n == 44100 && m < 1e-3, "48 kHz -> 44.1 kHz, 1 kHz sine", m, 1e-3);
    }

    printf(failures ? "FAIL: %d check(s)\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}

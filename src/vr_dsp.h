// VR's portable spectral pipeline. Zero-centred STFT padding, ordinary stereo,
// UVR 5.1 linear band filters, and scipy-compatible rational polyphase resampling.
#pragma once
#include "stft.h"
#include <algorithm>
#include <numeric>

namespace st {

// Bluestein extends the existing radix-2 FFT to UVR's 320/640/960 transforms.
class VRFFT {
    int n_, m_;
    FFT fft_;
    std::vector<cplx> chirp_, kernel_;
public:
    explicit VRFFT(int n) : n_(n), m_([&] { int m = 1; while (m < 2*n-1) m *= 2; return m; }()),
                           fft_(m_), chirp_(n), kernel_(m_) {
        for (int i = 0; i < n_; i++) {
            chirp_[i] = std::polar(1.0, -M_PI * (double)i * i / n_);
            kernel_[i] = std::conj(chirp_[i]);
            if (i) kernel_[m_-i] = kernel_[i];
        }
        fft_.run(kernel_, false);
    }
    void run(std::vector<cplx>& x, bool inverse) const {
        std::vector<cplx> a(m_);
        for (int i = 0; i < n_; i++) a[i] = (inverse ? std::conj(x[i]) : x[i]) * chirp_[i];
        fft_.run(a, false);
        for (int i = 0; i < m_; i++) a[i] *= kernel_[i];
        fft_.run(a, true);
        for (int i = 0; i < n_; i++) {
            cplx v = a[i] * chirp_[i] / (double)m_;
            x[i] = inverse ? std::conj(v) : v;
        }
    }
};

struct VRBand {
    int sr, hop, nfft, start, stop, hp_start, hp_stop, lp_start, lp_stop;
};

inline std::vector<cplx> vr_stft(const std::vector<float>& wave, int n, const VRBand& b, int& frames) {
    frames = n / b.hop + 1;
    const int F = b.nfft / 2 + 1;
    VRFFT fft(b.nfft);
    const auto win = hann_periodic(b.nfft);
    std::vector<cplx> out((size_t)2 * F * frames), buf(b.nfft);
    for (int c = 0; c < 2; c++)
        for (int t = 0; t < frames; t++) {
            for (int j = 0; j < b.nfft; j++) {
                const int k = t * b.hop + j - b.nfft / 2;
                buf[j] = k >= 0 && k < n ? wave[(size_t)c*n+k] * win[j] : 0.0;
            }
            fft.run(buf, false);
            for (int f = 0; f < F; f++) {
                const auto v = buf[f];
                // librosa's complex64 storage rounds before subsequent band operations.
                out[((size_t)c*F+f)*frames+t] = cplx((float)v.real(), (float)v.imag());
            }
        }
    return out;
}

inline std::vector<float> vr_istft(const std::vector<cplx>& spec, int frames, const VRBand& b) {
    const int F = b.nfft / 2 + 1, n = b.hop * (frames - 1);
    const size_t full = (size_t)b.nfft + n;
    VRFFT fft(b.nfft);
    const auto win = hann_periodic(b.nfft);
    std::vector<double> env(full);
    for (int t = 0; t < frames; t++)
        for (int j = 0; j < b.nfft; j++) env[(size_t)t*b.hop+j] += win[j]*win[j];
    std::vector<float> out((size_t)2*n);
    std::vector<cplx> buf(b.nfft);
    for (int c = 0; c < 2; c++) {
        std::vector<double> y(full);
        for (int t = 0; t < frames; t++) {
            for (int f = 0; f < F; f++) buf[f] = spec[((size_t)c*F+f)*frames+t];
            buf[0] = cplx(buf[0].real(), 0);
            buf[F-1] = cplx(buf[F-1].real(), 0);
            for (int f = 1; f < F-1; f++) buf[b.nfft-f] = std::conj(buf[f]);
            fft.run(buf, true);
            for (int j = 0; j < b.nfft; j++) y[(size_t)t*b.hop+j] += buf[j].real()/b.nfft*win[j];
        }
        for (int j = 0; j < n; j++) {
            const size_t k = (size_t)j+b.nfft/2;
            out[(size_t)c*n+j] = env[k] > 1e-11 ? (float)(y[k]/env[k]) : 0.0f;
        }
    }
    return out;
}

// Exactly scipy's centred upfirdn indexing, including ceil output length and zero edges.
inline std::vector<float> vr_resample(const std::vector<float>& x, int n, int src, int dst,
                                     const std::vector<float>& fir, int& out_n) {
    if (src == dst) { out_n = n; return x; }
    const int g = std::gcd(src, dst), up = dst/g, down = src/g;
    out_n = (int)(((int64_t)n*up+down-1)/down);
    const int half = (int)fir.size()/2;
    std::vector<float> y((size_t)2*out_n);
    for (int c = 0; c < 2; c++)
        for (int j = 0; j < out_n; j++) {
            const int64_t pos = (int64_t)j*down+half;
            const int lo = (int)std::max<int64_t>(0, (pos-(int)fir.size()+up)/up);
            const int hi = (int)std::min<int64_t>(n-1, pos/up);
            double sum = 0;
            for (int i = lo; i <= hi; i++) sum += (double)x[(size_t)c*n+i]*fir[(size_t)(pos-(int64_t)i*up)];
            y[(size_t)c*out_n+j] = (float)sum;
        }
    return y;
}

inline float vr_lowpass(int f, int start, int stop) {
    if (start < 0) return 1;
    if (f < start-1) return 1;
    if (f >= stop) return 0;
    return 1.0f - (float)(f-start+1)/(stop-start);
}
inline float vr_highpass(int f, int start, int stop) {
    if (start <= 0) return 1;
    // UVR passes hpf_stop-1 to get_hp_filter_mask.
    if (f <= stop+1) return 0;
    if (f >= start+1) return 1;
    return (float)(f-stop-1)/(start-stop);
}
} // namespace st

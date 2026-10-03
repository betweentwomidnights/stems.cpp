// roformer.cpp — see roformer.h. Layout conventions used throughout:
//
//   band features  [D, B, T]  (ggml ne; D fastest, then band, then frame) = torch's (t, band, d)
//   attention      q/k/v [dh, H, N, batch] for rope, [dh, N, H, batch] for the matmuls
//
// GGUF tensor names (written by tools/convert_roformer.py):
//   band.{b}.norm, band.{b}.weight, band.{b}.bias                 BandSplit, one per band
//   layer.{l}.{time,freq}.{j}.{attn_norm,qkv,gate.weight,gate.bias,out,
//                              ff_norm,ff1.weight,ff1.bias,ff2.weight,ff2.bias}
//   layer.{l}.{time,freq}.norm                                    when the Transformer norms its output
//   final_norm                                                    BS-RoFormer's norm after the layers
//   mask.{s}.{b}.{k}.weight, mask.{s}.{b}.{k}.bias                k < mask_layers, tanh between
#include "roformer.h"

#include "gguf_model.h"
#include "stft.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace st {

namespace {

// F.normalize(x) * sqrt(dim) is exactly x / sqrt(mean(x^2)); F.normalize's eps (1e-12 on the
// norm) only matters for an all-zero row, which this eps keeps at zero instead of NaN.
constexpr float kRmsEps = 1e-24f;
// Upper bound on one attention score tensor; the batch is split into groups to stay under it.
constexpr size_t kMaxScoreBytes = (size_t)256 << 20;

struct Graph {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor* in_feat = nullptr;          // band-major blocks of [dim_in_b, T]
    ggml_tensor* pos_t = nullptr;            // [T] rope positions for the time transformer
    ggml_tensor* pos_f = nullptr;            // [B] rope positions for the freq transformer
    std::vector<ggml_tensor*> masks;         // [stem * B + band] -> [dim_in_b, T]
    ~Graph() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
    }
};

ggml_tensor* concat_all(ggml_context* ctx, std::vector<ggml_tensor*> v, int dim) {
    while (v.size() > 1) {   // balanced, so each element is copied log2(n) times, not n
        std::vector<ggml_tensor*> next;
        for (size_t i = 0; i + 1 < v.size(); i += 2) next.push_back(ggml_concat(ctx, v[i], v[i + 1], dim));
        if (v.size() % 2) next.push_back(v.back());
        v.swap(next);
    }
    return v[0];
}

} // namespace

struct RoFormer::Impl {
    ggml_backend_t backend = nullptr;
    GgufModel model;
    std::string name, arch;
    std::vector<std::string> sources;        // model stems, then the complement if any
    int n_stems = 1;                         // the model's own stems
    bool complement = false;
    int sr = 44100, channels = 2;
    int dim = 0, depth = 0, time_depth = 1, freq_depth = 1, heads = 8, dim_head = 64, mask_layers = 3;
    int n_fft = 2048, hop = 441, chunk = 0, num_overlap = 2;
    bool stft_normalized = false;
    bool zero_dc = false;                   // zero the masked DC bin before the iSTFT (MSST)
    float rope_theta = 10000.0f;
    std::vector<int> band_sizes, band_freqs; // freqs per band, and their STFT bin indices
    std::vector<size_t> band_freq_off, band_feat_off;
    std::vector<int> bands_per_freq;         // [F] how many bands cover each bin
    std::unique_ptr<FFT> fft;

    int n_bins() const { return n_fft / 2 + 1; }
    int n_frames() const { return chunk / hop + 1; }
    int n_bands() const { return (int)band_sizes.size(); }
    int dim_in(int b) const { return band_sizes[b] * channels * 2; }
    size_t total_feat() const { return band_feat_off.back(); }

    ggml_tensor* W(const std::string& n) const { return model.get(n); }

    ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x, const std::string& w, const std::string& b = "") const {
        ggml_tensor* y = ggml_mul_mat(ctx, W(w), x);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        return b.empty() ? y : ggml_add(ctx, y, W(b));
    }

    ggml_tensor* rms(ggml_context* ctx, ggml_tensor* x, const std::string& gamma) const {
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, kRmsEps), W(gamma));
    }

    // Attention over N tokens for each of `batch` sequences. x: [D, N, batch].
    ggml_tensor* attention(ggml_context* ctx, ggml_tensor* x, const std::string& n, ggml_tensor* pos) const {
        const int64_t N = x->ne[1], batch = x->ne[2], H = heads, dh = dim_head;
        ggml_tensor* h = rms(ctx, x, n + ".attn_norm");
        ggml_tensor* qkv = linear(ctx, h, n + ".qkv");                     // [3*H*dh, N, batch]
        auto part = [&](int i) {                                           // 'b n (qkv h d)'
            return ggml_cont(ctx, ggml_view_4d(ctx, qkv, dh, H, N, batch, dh * sizeof(float),
                                               qkv->nb[1], qkv->nb[2], (size_t)i * H * dh * sizeof(float)));
        };
        auto rope = [&](ggml_tensor* t) {                                  // adjacent pairs (GPT-J)
            return ggml_rope_ext(ctx, t, pos, nullptr, (int)dh, 0, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        };
        ggml_tensor* q = ggml_cont(ctx, ggml_permute(ctx, rope(part(0)), 0, 2, 1, 3));   // [dh, N, H, batch]
        ggml_tensor* k = ggml_cont(ctx, ggml_permute(ctx, rope(part(1)), 0, 2, 1, 3));
        ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, part(2), 1, 2, 0, 3));        // [N, dh, H, batch]

        const size_t per = (size_t)N * N * H * sizeof(float);
        const int64_t group = std::max<int64_t>(1, std::min<int64_t>(batch, (int64_t)(kMaxScoreBytes / per)));
        std::vector<ggml_tensor*> outs;
        for (int64_t b0 = 0; b0 < batch; b0 += group) {
            const int64_t g = std::min(group, batch - b0);
            auto slice = [&](ggml_tensor* t) {
                return ggml_view_4d(ctx, t, t->ne[0], t->ne[1], t->ne[2], g, t->nb[1], t->nb[2], t->nb[3],
                                    (size_t)b0 * t->nb[3]);
            };
            ggml_tensor* kq = ggml_mul_mat(ctx, slice(k), slice(q));      // [N, N, H, g]
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / std::sqrt((float)dh), 0.0f);
            ggml_tensor* o = ggml_mul_mat(ctx, slice(vt), kq);              // [dh, N, H, g]
            ggml_mul_mat_set_prec(o, GGML_PREC_F32);
            outs.push_back(o);
        }
        ggml_tensor* o = concat_all(ctx, outs, 3);
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));              // [dh, H, N, batch]
        if (model.has(n + ".gate.weight")) {
            ggml_tensor* gates = ggml_sigmoid(ctx, linear(ctx, h, n + ".gate.weight", n + ".gate.bias"));
            o = ggml_mul(ctx, o, ggml_reshape_4d(ctx, gates, 1, H, N, batch));
        }
        return linear(ctx, ggml_reshape_3d(ctx, o, H * dh, N, batch), n + ".out");
    }

    ggml_tensor* transformer(ggml_context* ctx, ggml_tensor* x, const std::string& n, int layers,
                             ggml_tensor* pos) const {
        for (int j = 0; j < layers; j++) {
            const std::string l = n + "." + std::to_string(j);
            x = ggml_add(ctx, x, attention(ctx, x, l, pos));
            ggml_tensor* h = rms(ctx, x, l + ".ff_norm");
            h = ggml_gelu_erf(ctx, linear(ctx, h, l + ".ff1.weight", l + ".ff1.bias"));
            x = ggml_add(ctx, x, linear(ctx, h, l + ".ff2.weight", l + ".ff2.bias"));
        }
        return model.has(n + ".norm") ? rms(ctx, x, n + ".norm") : x;
    }

    std::unique_ptr<Graph> build() const {
        auto g = std::make_unique<Graph>();
        const size_t n_nodes = 16384;
        ggml_init_params ip = {ggml_tensor_overhead() * n_nodes * 2 + ggml_graph_overhead_custom(n_nodes, false),
                               nullptr, true};
        g->ctx = ggml_init(ip);
        ggml_context* ctx = g->ctx;
        const int T = n_frames(), B = n_bands(), D = dim;

        g->in_feat = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)total_feat() * T);
        g->pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        g->pos_f = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, B);
        ggml_set_input(g->in_feat);
        ggml_set_input(g->pos_t);
        ggml_set_input(g->pos_f);

        // BandSplit: per band RMSNorm + Linear(dim_in_b -> D), stacked on the band axis.
        std::vector<ggml_tensor*> bands;
        for (int b = 0; b < B; b++) {
            const std::string n = "band." + std::to_string(b);
            ggml_tensor* v = ggml_view_2d(ctx, g->in_feat, dim_in(b), T, (size_t)dim_in(b) * sizeof(float),
                                          band_feat_off[b] * T * sizeof(float));
            v = linear(ctx, rms(ctx, v, n + ".norm"), n + ".weight", n + ".bias");   // [D, T]
            bands.push_back(ggml_reshape_3d(ctx, v, D, 1, T));
        }
        ggml_tensor* x = concat_all(ctx, bands, 1);                        // [D, B, T]

        for (int l = 0; l < depth; l++) {
            const std::string n = "layer." + std::to_string(l);
            ggml_tensor* xt = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));          // [D, T, B]
            xt = transformer(ctx, xt, n + ".time", time_depth, g->pos_t);
            x = ggml_cont(ctx, ggml_permute(ctx, xt, 0, 2, 1, 3));                       // [D, B, T]
            x = transformer(ctx, x, n + ".freq", freq_depth, g->pos_f);
        }
        if (model.has("final_norm")) x = rms(ctx, x, "final_norm");

        // Mask estimators: per stem and band, an MLP (tanh between layers) then GLU.
        ggml_tensor* xb = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));              // [D, T, B]
        for (int s = 0; s < n_stems; s++) {
            for (int b = 0; b < B; b++) {
                const std::string n = "mask." + std::to_string(s) + "." + std::to_string(b) + ".";
                ggml_tensor* h = ggml_view_2d(ctx, xb, D, T, xb->nb[1], (size_t)b * xb->nb[2]);
                for (int k = 0; k < mask_layers; k++) {
                    h = linear(ctx, h, n + std::to_string(k) + ".weight", n + std::to_string(k) + ".bias");
                    if (k < mask_layers - 1) h = ggml_tanh(ctx, h);
                }
                const int64_t half = dim_in(b);                                          // GLU on dim 0
                ggml_tensor* a = ggml_view_2d(ctx, h, half, T, h->nb[1], 0);
                ggml_tensor* gate = ggml_view_2d(ctx, h, half, T, h->nb[1], half * sizeof(float));
                ggml_tensor* m = ggml_mul(ctx, a, ggml_sigmoid(ctx, gate));              // [dim_in_b, T]
                ggml_set_output(m);
                g->masks.push_back(m);
            }
        }

        g->gf = ggml_new_graph_custom(ctx, n_nodes, false);
        for (ggml_tensor* m : g->masks) ggml_build_forward_expand(g->gf, m);
        g->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(g->alloc, g->gf))
            throw std::runtime_error("roformer: failed to allocate the compute graph");
        return g;
    }

    // MelBandRoformer.forward / BSRoformer.forward on one chunk: [C][chunk] -> [S][C][chunk].
    void run(const Graph& g, const float* in, float* out) const {
        const int C = channels, L = chunk, T = n_frames(), F = n_bins(), B = n_bands();

        std::vector<std::vector<cplx>> z(C);
        for (int c = 0; c < C; c++) {
            int frames = 0;
            z[c] = torch_stft(*fft, in + (size_t)c * L, L, hop, stft_normalized, frames);
        }

        // 'b (f s) t c' indexed by the band frequency lists, then 'b f t c -> b t (f c)'.
        std::vector<float> feat(total_feat() * T);
        for (int b = 0; b < B; b++) {
            float* blk = &feat[band_feat_off[b] * T];
            const int din = dim_in(b);
            for (int j = 0; j < band_sizes[b]; j++) {
                const int f = band_freqs[band_freq_off[b] + j];
                for (int c = 0; c < C; c++)
                    for (int t = 0; t < T; t++) {
                        const cplx v = z[c][(size_t)f * T + t];
                        float* p = &blk[(size_t)t * din + (j * C + c) * 2];
                        p[0] = (float)v.real();
                        p[1] = (float)v.imag();
                    }
            }
        }
        std::vector<int32_t> pt(T), pf(B);
        for (int i = 0; i < T; i++) pt[i] = i;
        for (int i = 0; i < B; i++) pf[i] = i;
        ggml_backend_tensor_set(g.in_feat, feat.data(), 0, feat.size() * sizeof(float));
        ggml_backend_tensor_set(g.pos_t, pt.data(), 0, pt.size() * sizeof(int32_t));
        ggml_backend_tensor_set(g.pos_f, pf.data(), 0, pf.size() * sizeof(int32_t));
        std::string err;
        if (!graph_compute_checked(backend, g.gf, "roformer", err)) throw std::runtime_error(err);

        // Average the masks where bands overlap, apply them, and invert.
        std::vector<float> m;
        std::vector<cplx> acc((size_t)C * F * T), spec((size_t)F * T);
        for (int s = 0; s < n_stems; s++) {
            std::fill(acc.begin(), acc.end(), cplx(0.0, 0.0));
            for (int b = 0; b < B; b++) {
                ggml_tensor* mt = g.masks[(size_t)s * B + b];
                m.resize(ggml_nelements(mt));
                ggml_backend_tensor_get(mt, m.data(), 0, m.size() * sizeof(float));
                const int din = dim_in(b);
                for (int j = 0; j < band_sizes[b]; j++) {
                    const int f = band_freqs[band_freq_off[b] + j];
                    for (int c = 0; c < C; c++)
                        for (int t = 0; t < T; t++) {
                            const float* p = &m[(size_t)t * din + (j * C + c) * 2];
                            acc[((size_t)c * F + f) * T + t] += cplx(p[0], p[1]);
                        }
                }
            }
            for (int c = 0; c < C; c++) {
                for (int f = 0; f < F; f++) {
                    const double denom = std::max(bands_per_freq[f], 1);
                    for (int t = 0; t < T; t++) {
                        const size_t i = (size_t)f * T + t;
                        spec[i] = z[c][i] * (acc[(size_t)c * F * T + i] / denom);
                    }
                }
                if (zero_dc) std::fill(spec.begin(), spec.begin() + T, cplx(0.0, 0.0));
                const std::vector<float> y = torch_istft(*fft, spec, T, hop, stft_normalized);
                std::copy(y.begin(), y.begin() + L, out + ((size_t)s * C + c) * L);
            }
        }
    }

    // utils.demix_track: chunks of `chunk` every `step`, linear fades, reflect-padded borders.
    std::vector<float> demix(const Graph& g, const std::vector<float>& mix, int len, int step,
                             const std::function<void()>& tick) const {
        const int C = channels, S = n_stems, Cc = chunk;
        const int fade = Cc / 10, border = Cc - step;
        const bool pad = len > 2 * border && border > 0;
        const int Lp = pad ? len + 2 * border : len;
        std::vector<float> x((size_t)C * Lp);
        for (int c = 0; c < C; c++) {
            const float* src = &mix[(size_t)c * len];
            if (pad) {
                const std::vector<double> p = reflect_pad(src, len, border, border);
                std::transform(p.begin(), p.end(), &x[(size_t)c * Lp], [](double v) { return (float)v; });
            } else {
                std::copy(src, src + len, &x[(size_t)c * Lp]);
            }
        }

        // torch.linspace(0, 1, fade) in float32, computed the way torch does (from both ends).
        std::vector<float> fade_in(fade), window(Cc, 1.0f);
        if (fade > 1) {
            const float stp = 1.0f / (float)(fade - 1);
            for (int i = 0; i < fade; i++)
                fade_in[i] = i < fade / 2 ? (float)i * stp : 1.0f - (float)(fade - 1 - i) * stp;
        }
        for (int i = 0; i < fade; i++) {
            window[i] *= fade_in[i];
            window[Cc - 1 - i] *= fade_in[i];
        }

        std::vector<double> result((size_t)S * C * Lp, 0.0), counter(Lp, 0.0);
        std::vector<float> part((size_t)C * Cc), y((size_t)S * C * Cc);
        for (int i = 0; i < Lp; i += step) {
            const int length = std::min(Cc, Lp - i);
            for (int c = 0; c < C; c++) {
                const float* src = &x[(size_t)c * Lp + i];
                float* dst = &part[(size_t)c * Cc];
                std::copy(src, src + length, dst);
                if (length < Cc) {
                    if (length > Cc / 2 + 1) {
                        const std::vector<double> p = reflect_pad(src, length, 0, Cc - length);
                        std::transform(p.begin(), p.end(), dst, [](double v) { return (float)v; });
                    } else {
                        std::fill(dst + length, dst + Cc, 0.0f);
                    }
                }
            }
            tick();
            run(g, part.data(), y.data());
            std::vector<float> w = window;
            if (i == 0) std::fill(w.begin(), w.begin() + fade, 1.0f);
            else if (i + Cc >= Lp) std::fill(w.end() - fade, w.end(), 1.0f);
            for (int k = 0; k < S * C; k++)
                for (int t = 0; t < length; t++)
                    result[(size_t)k * Lp + i + t] += (double)y[(size_t)k * Cc + t] * w[t];
            for (int t = 0; t < length; t++) counter[i + t] += w[t];
        }

        std::vector<float> out((size_t)S * C * len);
        const int off = pad ? border : 0;
        for (int k = 0; k < S * C; k++)
            for (int t = 0; t < len; t++) {
                const double v = result[(size_t)k * Lp + off + t] / counter[off + t];
                out[(size_t)k * len + t] = std::isfinite(v) ? (float)v : 0.0f;   // np.nan_to_num
            }
        return out;
    }

    int n_chunks(int len, int step) const {
        const int border = chunk - step;
        const int Lp = (len > 2 * border && border > 0) ? len + 2 * border : len;
        return (Lp + step - 1) / step;
    }
};

RoFormer::RoFormer(const std::string& path, const char* device, int cpu_threads) : p_(new Impl) {
    Impl& P = *p_;
    P.backend = make_backend(cpu_threads, device);
    P.model = load_gguf(path.c_str(), P.backend);
    const GgufModel& m = P.model;
    gguf_context* gg = m.gguf;
    P.arch = m.string("general.architecture");
    if (P.arch != "mel_band_roformer" && P.arch != "bs_roformer")
        throw std::runtime_error(path + ": not a roformer GGUF");
    P.name = m.string("stems.model");
    const int ks = gguf_find_key(gg, "stems.sources");
    if (ks < 0) throw std::runtime_error(path + ": missing stems.sources");
    for (size_t i = 0; i < gguf_get_arr_n(gg, ks); i++) P.sources.push_back(gguf_get_arr_str(gg, ks, i));
    P.n_stems = (int)P.sources.size();
    if (gguf_find_key(gg, "stems.complement") >= 0) {
        P.complement = true;
        P.sources.push_back(m.string("stems.complement"));
    }
    P.sr = (int)m.u32("stems.samplerate");
    P.channels = (int)m.u32("stems.audio_channels");
    P.dim = (int)m.u32("roformer.dim");
    P.depth = (int)m.u32("roformer.depth");
    P.time_depth = (int)m.u32("roformer.time_transformer_depth");
    P.freq_depth = (int)m.u32("roformer.freq_transformer_depth");
    P.heads = (int)m.u32("roformer.heads");
    P.dim_head = (int)m.u32("roformer.dim_head");
    P.mask_layers = (int)m.u32("roformer.mask_layers");
    P.n_fft = (int)m.u32("roformer.stft_n_fft");
    P.hop = (int)m.u32("roformer.stft_hop_length");
    P.stft_normalized = m.boolean("roformer.stft_normalized");
    P.zero_dc = m.boolean("roformer.zero_dc");
    P.rope_theta = m.f32("roformer.rope_theta");
    P.chunk = (int)m.u32("roformer.chunk_size");
    P.num_overlap = (int)m.u32("roformer.num_overlap");
    for (int32_t v : m.i32s("roformer.band_sizes")) P.band_sizes.push_back(v);
    for (int32_t v : m.i32s("roformer.band_freqs")) P.band_freqs.push_back(v);
    if (P.chunk % P.hop) throw std::runtime_error(path + ": chunk_size is not a multiple of the hop");

    P.fft = std::make_unique<FFT>(P.n_fft);
    P.bands_per_freq.assign(P.n_bins(), 0);
    P.band_freq_off.push_back(0);
    P.band_feat_off.push_back(0);
    for (int b = 0; b < P.n_bands(); b++) {
        P.band_freq_off.push_back(P.band_freq_off.back() + P.band_sizes[b]);
        P.band_feat_off.push_back(P.band_feat_off.back() + P.dim_in(b));
    }
    if (P.band_freq_off.back() != P.band_freqs.size())
        throw std::runtime_error(path + ": roformer.band_sizes does not match roformer.band_freqs");
    for (int f : P.band_freqs) {
        if (f < 0 || f >= P.n_bins()) throw std::runtime_error(path + ": band frequency out of range");
        P.bands_per_freq[f]++;
    }
}

RoFormer::~RoFormer() {
    if (p_) {
        p_->model.free();
        if (p_->backend) ggml_backend_free(p_->backend);
    }
}

const std::string& RoFormer::name() const { return p_->name; }
const std::string& RoFormer::architecture() const { return p_->arch; }
const std::vector<std::string>& RoFormer::sources() const { return p_->sources; }
int RoFormer::samplerate() const { return p_->sr; }
int RoFormer::audio_channels() const { return p_->channels; }
int RoFormer::segment_samples() const { return p_->chunk; }
const char* RoFormer::backend_name() const { return ggml_backend_name(p_->backend); }

std::vector<float> RoFormer::forward(const float* in) const {
    auto g = p_->build();
    std::vector<float> out((size_t)p_->n_stems * p_->channels * p_->chunk);
    p_->run(*g, in, out.data());
    return out;
}

std::vector<float> RoFormer::separate(const float* mix_in, int len, const SeparateOptions& opt) const {
    const Impl& P = *p_;
    const int C = P.channels, S = P.n_stems;
    if (len <= 0) throw std::invalid_argument("empty input");
    if (opt.overlap >= 1.0f) throw std::invalid_argument("overlap must be < 1");
    // Default: the checkpoint's own num_overlap, with demix_track's integer step.
    const int step = opt.overlap < 0.0f ? P.chunk / P.num_overlap
                                        : std::max(1, (int)std::lround(P.chunk * (1.0 - opt.overlap)));
    const int passes = std::max(1, opt.shifts);
    const int total = passes * P.n_chunks(len, step);
    int done = 0;
    auto tick = [&]() {
        if (opt.should_cancel && opt.should_cancel()) throw Cancelled();
        if (opt.progress && done > 0) opt.progress(done, total);
        done++;
    };

    auto g = P.build();
    const std::vector<float> mix(mix_in, mix_in + (size_t)C * len);
    std::vector<float> est;
    if (opt.shifts <= 0) {
        est = P.demix(*g, mix, len, step, tick);
    } else {
        // demucs' shift trick: average over random offsets into a zero-padded copy.
        const int max_shift = P.sr / 2;
        std::mt19937 rng(opt.seed);
        std::uniform_int_distribution<int> dist(0, max_shift - 1);
        est.assign((size_t)S * C * len, 0.0f);
        for (int k = 0; k < opt.shifts; k++) {
            const int offset = dist(rng);
            const int n = len + max_shift - offset;
            std::vector<float> shifted((size_t)C * n, 0.0f);
            for (int c = 0; c < C; c++)
                for (int t = 0; t < n; t++) {
                    const int src = t + offset - max_shift;
                    if (src >= 0 && src < len) shifted[(size_t)c * n + t] = mix[(size_t)c * len + src];
                }
            const std::vector<float> y = P.demix(*g, shifted, n, step, tick);
            for (int i = 0; i < S * C; i++)
                for (int t = 0; t < len; t++)
                    est[(size_t)i * len + t] += y[(size_t)i * n + max_shift - offset + t] / opt.shifts;
        }
    }
    if (opt.progress) opt.progress(total, total);

    if (!P.complement) return est;
    std::vector<float> out((size_t)(S + 1) * C * len);
    std::copy(est.begin(), est.end(), out.begin());
    for (size_t i = 0; i < (size_t)C * len; i++) out[(size_t)S * C * len + i] = mix[i] - est[i];
    return out;
}

} // namespace st

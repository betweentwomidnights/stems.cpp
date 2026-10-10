// tfc_tdf.cpp — see tfc_tdf.h and docs/MDX.md. Layout conventions used throughout:
//
//   spectrogram  [T, F, 4 * stems]  (ggml ne) = torch's (stems * 4, F, T); the 4 channels per
//                                   stem are left real, left imaginary, right real, right imag
//   inside the U-Net  [F, T, C]     = torch's (C, T, F) after the model's transpose(-1, -2), so
//                                   the frequency Linears (TDF) are plain matmuls over ne0
//
// GGUF tensor names (written by tools/convert_mdx.py), variant mdx_net (BN folded by the ONNX
// exporter or the converter, so every convolution has a bias):
//   first.{weight,bias}                       1x1, then ReLU, then transpose
//   {enc.i,mid,dec.i}.tfc.j.{weight,bias}     3x3 + ReLU, j < num_blocks
//   {enc.i,mid,dec.i}.tdf{1,2}.weight         Linear over frequency, no bias
//   {enc.i,mid,dec.i}.tdf{1,2}.{scale,shift}  the BatchNorm after each Linear, then ReLU
//   enc.i.down.{weight,bias}                  2x2 stride 2 + ReLU
//   dec.i.up.{weight,bias}                    2x2 stride 2 transposed + ReLU, then x * skip
//   final.{weight,bias}                       1x1 after transposing back
// Variant mdx23c keeps the PyTorch names of ZFTurbo's TFC_TDF_net (models/mdx23c_tfc_tdf_v3.py).
// Transposed convolution weights are [in, kw, kh, out] in both.
#include "tfc_tdf.h"

#include "gguf_model.h"
#include "stft.h"
#include "vr_dsp.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace st {

namespace {

constexpr float kNormEps = 1e-5f;      // nn.InstanceNorm2d's default
constexpr size_t kMaxNodes = 8192;
constexpr size_t kIm2colFloats = (size_t)32 << 20;   // 128 MB per 3x3 im2col band

struct Graph {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor* input = nullptr;      // [T, F, 4]
    ggml_tensor* output = nullptr;     // [T, F, 4 * stems]
    ~Graph() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
    }
};

// torch.linspace in float32, computed the way torch does (from both ends).
std::vector<float> linspace(float a, float b, int n) {
    std::vector<float> v(std::max(n, 0));
    if (n == 1) v[0] = a;
    if (n < 2) return v;
    const float step = (b - a) / (float)(n - 1);
    for (int i = 0; i < n; i++) v[i] = i < n / 2 ? a + step * (float)i : b - step * (float)(n - 1 - i);
    return v;
}

// np.hanning: the symmetric window.
std::vector<double> hanning(int n) {
    std::vector<double> w(n, 1.0);
    if (n > 1)
        for (int i = 0; i < n; i++) w[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / (n - 1));
    return w;
}

} // namespace

struct TFCTDF::Impl {
    ggml_backend_t backend = nullptr;
    GgufModel model;
    std::string name, arch = "tfc_tdf", variant, chunking;
    std::vector<std::string> sources;
    int n_stems = 1;
    bool complement = false;
    int sr = 44100, n_fft = 0, hop = 0, dim_f = 0, dim_t = 0, subbands = 1, scales = 0, blocks = 0;
    int channels = 0, growth = 0, bottleneck = 0, zero_bins = 0, num_overlap = 0;
    bool gelu = false;
    float compensate = 1.0f;
    std::unique_ptr<FFT> fft2;         // power-of-two n_fft
    std::unique_ptr<VRFFT> fftb;       // anything else (Kim_Vocal_2's 7680)

    ~Impl() {
        model.free();
        if (backend) ggml_backend_free(backend);
    }

    bool mdx_net() const { return variant == "mdx_net"; }
    int chunk() const { return hop * (dim_t - 1); }
    int in_channels() const { return 4 * subbands; }
    ggml_tensor* W(const std::string& n) const { return model.get(n); }

    // ---- weight contract (mirrors expected_shapes() in tools/convert_mdx.py) -------------------

    void want(const std::string& n, std::vector<int64_t> torch_shape) const {
        ggml_tensor* t = W(n);
        std::reverse(torch_shape.begin(), torch_shape.end());
        torch_shape.resize(4, 1);
        if (t->type != GGML_TYPE_F32 || t->ne[0] != torch_shape[0] || t->ne[1] != torch_shape[1] ||
            t->ne[2] != torch_shape[2] || t->ne[3] != torch_shape[3])
            throw std::runtime_error("tfc_tdf: invalid F32 tensor shape: " + n);
    }

    void validate_weights() const {
        const int64_t l = blocks, g = growth, bn = bottleneck, cin = in_channels(), S = n_stems;
        int64_t c = channels, f = dim_f / subbands;
        if (mdx_net()) {
            auto block = [&](const std::string& p, int64_t c, int64_t f) {
                for (int j = 0; j < l; j++) {
                    want(p + ".tfc." + std::to_string(j) + ".weight", {c, c, 3, 3});
                    want(p + ".tfc." + std::to_string(j) + ".bias", {c, 1, 1});
                }
                want(p + ".tdf1.weight", {f / bn, f});
                want(p + ".tdf2.weight", {f, f / bn});
                for (const char* k : {".tdf1.scale", ".tdf1.shift", ".tdf2.scale", ".tdf2.shift"}) want(p + k, {c, 1, 1});
            };
            want("first.weight", {c, cin, 1, 1});
            want("first.bias", {c, 1, 1});
            for (int i = 0; i < scales; i++) {
                const std::string p = "enc." + std::to_string(i);
                block(p, c, f);
                want(p + ".down.weight", {c + g, c, 2, 2});
                want(p + ".down.bias", {c + g, 1, 1});
                c += g; f /= 2;
            }
            block("mid", c, f);
            for (int i = 0; i < scales; i++) {
                const std::string p = "dec." + std::to_string(i);
                want(p + ".up.weight", {c - g, 2, 2, c});
                want(p + ".up.bias", {c - g, 1, 1});
                c -= g; f *= 2;
                block(p, c, f);
            }
            want("final.weight", {cin * S, c, 1, 1});
            want("final.bias", {cin * S, 1, 1});
        } else {
            auto block = [&](const std::string& p, int64_t cin_b, int64_t c, int64_t f) {
                for (int j = 0; j < l; j++) {
                    const std::string q = p + ".blocks." + std::to_string(j);
                    const int64_t ci = j == 0 ? cin_b : c;
                    want(q + ".tfc1.0.weight", {ci, 1, 1});
                    want(q + ".tfc1.0.bias", {ci, 1, 1});
                    want(q + ".tfc1.2.weight", {c, ci, 3, 3});
                    for (const char* k : {".tdf.0", ".tdf.3", ".tfc2.0"}) {
                        want(q + k + ".weight", {c, 1, 1});
                        want(q + k + ".bias", {c, 1, 1});
                    }
                    want(q + ".tdf.2.weight", {f / bn, f});
                    want(q + ".tdf.5.weight", {f, f / bn});
                    want(q + ".tfc2.2.weight", {c, c, 3, 3});
                    want(q + ".shortcut.weight", {c, ci, 1, 1});
                }
            };
            want("first_conv.weight", {c, cin, 1, 1});
            for (int i = 0; i < scales; i++) {
                const std::string p = "encoder_blocks." + std::to_string(i);
                block(p + ".tfc_tdf", c, c, f);
                want(p + ".downscale.conv.0.weight", {c, 1, 1});
                want(p + ".downscale.conv.0.bias", {c, 1, 1});
                want(p + ".downscale.conv.2.weight", {c + g, c, 2, 2});
                c += g; f /= 2;
            }
            block("bottleneck_block", c, c, f);
            for (int i = 0; i < scales; i++) {
                const std::string p = "decoder_blocks." + std::to_string(i);
                want(p + ".upscale.conv.0.weight", {c, 1, 1});
                want(p + ".upscale.conv.0.bias", {c, 1, 1});
                want(p + ".upscale.conv.2.weight", {c - g, 2, 2, c});
                c -= g; f *= 2;
                block(p + ".tfc_tdf", 2 * c, c, f);
            }
            want("final_conv.0.weight", {c, c + cin, 1, 1});
            want("final_conv.2.weight", {S * cin, c, 1, 1});
        }
    }

    // ---- graph pieces ---------------------------------------------------------------------

    static ggml_tensor* mm(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
        ggml_tensor* y = ggml_mul_mat(ctx, a, b);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        return y;
    }

    // im2col + F32 matmul. ggml's direct CONV_2D matches too since betweentwomidnights/ggml#16
    // (before it, Vulkan ran it in fp16 with cooperative matrices: Kim at 29 dB), but it is slower
    // here on CUDA (5-12x) and CPU, and no faster on Vulkan; docs/MDX.md has the timings.
    ggml_tensor* im2col_mm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* k, int stride, int p0, int p1) const {
        ggml_tensor* im = ggml_im2col(ctx, k, x, stride, stride, p0, p1, 1, 1, true, GGML_TYPE_F32);
        ggml_tensor* y = mm(ctx, ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1] * im->ne[2]),
                            ggml_reshape_2d(ctx, k, k->ne[0] * k->ne[1] * k->ne[2], k->ne[3]));
        return ggml_reshape_3d(ctx, y, im->ne[1], im->ne[2], k->ne[3]);
    }

    // x [W, H, Cin] -> [W', H', Cout]. 1x1 and 2x2 stride 2 are non-overlapping, so their im2col
    // is no larger than the input. A 3x3 im2col is nine times the input (1.4 GB at Kim's first
    // scale), so it runs in bands of output rows, each with its one-row halo, under kIm2colFloats.
    ggml_tensor* conv(ggml_context* ctx, ggml_tensor* x, const std::string& n, int stride = 1, int pad = 0,
                      bool bias = false) const {
        ggml_tensor* k = W(n + ".weight");
        if (k->ne[2] != x->ne[2]) throw std::runtime_error("tfc_tdf: convolution input mismatch: " + n);
        ggml_tensor* y;
        const int64_t Wd = x->ne[0], H = x->ne[1], row = k->ne[0] * k->ne[1] * k->ne[2] * Wd;
        const int64_t band = std::max<int64_t>(1, (int64_t)kIm2colFloats / row);
        if (k->ne[0] != 3 || stride != 1 || pad != 1 || band >= H) {
            y = im2col_mm(ctx, x, k, stride, pad, pad);
        } else {
            // Zero rows above and below, as pad-right-then-roll: Metal's PAD only pads on the right.
            ggml_tensor* xp = ggml_roll(ctx, ggml_pad(ctx, x, 0, 2, 0, 0), 0, 1, 0, 0);
            std::vector<ggml_tensor*> parts;
            for (int64_t h0 = 0; h0 < H; h0 += band) {
                const int64_t rows = std::min(band, H - h0);
                ggml_tensor* v = ggml_view_3d(ctx, xp, Wd, rows + 2, xp->ne[2], xp->nb[1], xp->nb[2], h0 * xp->nb[1]);
                parts.push_back(im2col_mm(ctx, ggml_cont(ctx, v), k, 1, 1, 0));
            }
            while (parts.size() > 1) {    // balanced, so each band is copied log2(n) times
                std::vector<ggml_tensor*> next;
                for (size_t i = 0; i + 1 < parts.size(); i += 2) next.push_back(ggml_concat(ctx, parts[i], parts[i + 1], 1));
                if (parts.size() % 2) next.push_back(parts.back());
                parts.swap(next);
            }
            y = parts[0];
        }
        return bias ? ggml_add(ctx, y, W(n + ".bias")) : y;
    }

    // ConvTranspose2d with stride == kernel (no overlap): one matmul over input channels, then
    // two permutes that interleave each pixel's kernel taps into the upsampled grid.
    ggml_tensor* conv_t(ggml_context* ctx, ggml_tensor* x, const std::string& n, bool bias) const {
        ggml_tensor* w = W(n + ".weight");                        // [Cin, KW, KH, Cout]
        const int64_t Wd = x->ne[0], H = x->ne[1], Ci = x->ne[2], K = w->ne[1], Co = w->ne[3];
        if (w->ne[0] != Ci) throw std::runtime_error("tfc_tdf: transposed convolution input mismatch: " + n);
        ggml_tensor* xc = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));                  // [Ci, W, H]
        ggml_tensor* y = mm(ctx, ggml_reshape_2d(ctx, w, Ci, K * K * Co), ggml_reshape_2d(ctx, xc, Ci, Wd * H));
        y = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, y, K, K * Co, Wd, H), 0, 2, 1, 3)); // [K, W, K*Co, H]
        y = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, y, K * Wd, K, Co, H), 0, 1, 3, 2)); // [K*W, K, H, Co]
        y = ggml_reshape_3d(ctx, y, K * Wd, K * H, Co);
        return bias ? ggml_add(ctx, y, W(n + ".bias")) : y;
    }

    // InstanceNorm2d(affine): every channel normalized over its whole [W, H] plane.
    ggml_tensor* norm(ggml_context* ctx, ggml_tensor* x, const std::string& n) const {
        const int64_t Wd = x->ne[0], H = x->ne[1], C = x->ne[2];
        ggml_tensor* y = ggml_norm(ctx, ggml_reshape_2d(ctx, x, Wd * H, C), kNormEps);
        y = ggml_reshape_3d(ctx, y, Wd, H, C);
        return ggml_add(ctx, ggml_mul(ctx, y, W(n + ".weight")), W(n + ".bias"));
    }

    ggml_tensor* act(ggml_context* ctx, ggml_tensor* x) const {
        return gelu ? ggml_gelu_erf(ctx, x) : ggml_relu(ctx, x);
    }

    ggml_tensor* tfc_tdf_v2(ggml_context* ctx, ggml_tensor* x, const std::string& p) const {
        for (int j = 0; j < blocks; j++) x = ggml_relu(ctx, conv(ctx, x, p + ".tfc." + std::to_string(j), 1, 1, true));
        ggml_tensor* y = x;
        for (int j = 1; j <= 2; j++) {
            const std::string q = p + ".tdf" + std::to_string(j);
            y = mm(ctx, W(q + ".weight"), y);
            y = ggml_relu(ctx, ggml_add(ctx, ggml_mul(ctx, y, W(q + ".scale")), W(q + ".shift")));
        }
        return ggml_add(ctx, x, y);
    }

    ggml_tensor* tfc_tdf_v3(ggml_context* ctx, ggml_tensor* x, const std::string& p) const {
        for (int j = 0; j < blocks; j++) {
            const std::string q = p + ".blocks." + std::to_string(j);
            ggml_tensor* s = conv(ctx, x, q + ".shortcut");
            x = conv(ctx, act(ctx, norm(ctx, x, q + ".tfc1.0")), q + ".tfc1.2", 1, 1);
            ggml_tensor* y = mm(ctx, W(q + ".tdf.2.weight"), act(ctx, norm(ctx, x, q + ".tdf.0")));
            y = mm(ctx, W(q + ".tdf.5.weight"), act(ctx, norm(ctx, y, q + ".tdf.3")));
            x = ggml_add(ctx, x, y);
            x = conv(ctx, act(ctx, norm(ctx, x, q + ".tfc2.0")), q + ".tfc2.2", 1, 1);
            x = ggml_add(ctx, x, s);
        }
        return x;
    }

    static ggml_tensor* transpose(ggml_context* ctx, ggml_tensor* x) {
        return ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    }

    // ConvTDFNet.forward (UVR lib_v5/mdxnet.py), as the ONNX graph records it.
    ggml_tensor* net_v2(ggml_context* ctx, ggml_tensor* x) const {
        x = transpose(ctx, ggml_relu(ctx, conv(ctx, x, "first", 1, 0, true)));
        std::vector<ggml_tensor*> skips;
        for (int i = 0; i < scales; i++) {
            const std::string p = "enc." + std::to_string(i);
            x = tfc_tdf_v2(ctx, x, p);
            skips.push_back(x);
            x = ggml_relu(ctx, conv(ctx, x, p + ".down", 2, 0, true));
        }
        x = tfc_tdf_v2(ctx, x, "mid");
        for (int i = 0; i < scales; i++) {
            const std::string p = "dec." + std::to_string(i);
            x = ggml_mul(ctx, ggml_relu(ctx, conv_t(ctx, x, p + ".up", true)), skips.back());
            skips.pop_back();
            x = tfc_tdf_v2(ctx, x, p);
        }
        return conv(ctx, transpose(ctx, x), "final", 1, 0, true);
    }

    // TFC_TDF_net.forward between its STFT and iSTFT.
    ggml_tensor* net_v3(ggml_context* ctx, ggml_tensor* x) const {
        const int64_t T = x->ne[0], Fs = dim_f / subbands, k = subbands, cin = in_channels();
        ggml_tensor* mix = ggml_reshape_3d(ctx, x, T, Fs, cin);              // cac2cws
        ggml_tensor* first = conv(ctx, mix, "first_conv");
        x = transpose(ctx, first);
        std::vector<ggml_tensor*> skips;
        for (int i = 0; i < scales; i++) {
            const std::string p = "encoder_blocks." + std::to_string(i);
            x = tfc_tdf_v3(ctx, x, p + ".tfc_tdf");
            skips.push_back(x);
            x = conv(ctx, act(ctx, norm(ctx, x, p + ".downscale.conv.0")), p + ".downscale.conv.2", 2, 0);
        }
        x = tfc_tdf_v3(ctx, x, "bottleneck_block");
        for (int i = 0; i < scales; i++) {
            const std::string p = "decoder_blocks." + std::to_string(i);
            x = conv_t(ctx, act(ctx, norm(ctx, x, p + ".upscale.conv.0")), p + ".upscale.conv.2", false);
            x = ggml_concat(ctx, x, skips.back(), 2);
            skips.pop_back();
            x = tfc_tdf_v3(ctx, x, p + ".tfc_tdf");
        }
        x = ggml_mul(ctx, transpose(ctx, x), first);                         // "reduce artifacts"
        x = conv(ctx, act(ctx, conv(ctx, ggml_concat(ctx, mix, x, 2), "final_conv.0")), "final_conv.2");
        return ggml_reshape_3d(ctx, x, T, Fs * k, x->ne[2] / k);              // cws2cac
    }

    std::unique_ptr<Graph> build() const {
        auto g = std::make_unique<Graph>();
        ggml_init_params init = {kMaxNodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(kMaxNodes, false),
                                 nullptr, true};
        g->ctx = ggml_init(init);
        if (!g->ctx) throw std::runtime_error("tfc_tdf: graph context allocation failed");
        g->input = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F32, dim_t, dim_f, 4);
        ggml_set_input(g->input);
        g->output = ggml_cont(g->ctx, mdx_net() ? net_v2(g->ctx, g->input) : net_v3(g->ctx, g->input));
        ggml_set_output(g->output);
        if (g->output->ne[0] != dim_t || g->output->ne[1] != dim_f || g->output->ne[2] != 4 * n_stems)
            throw std::runtime_error("tfc_tdf: unexpected network output shape");
        g->gf = ggml_new_graph_custom(g->ctx, kMaxNodes, false);
        ggml_build_forward_expand(g->gf, g->output);
        g->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(g->alloc, g->gf)) throw std::runtime_error("tfc_tdf: graph allocation failed");
        return g;
    }

    // ---- one chunk: STFT -> network -> iSTFT ------------------------------------------------

    std::vector<cplx> stft(const float* x, int& T) const {
        return fft2 ? torch_stft(*fft2, x, chunk(), hop, false, T) : torch_stft(*fftb, x, chunk(), hop, false, T);
    }
    std::vector<float> istft(const std::vector<cplx>& s, int T) const {
        return fft2 ? torch_istft(*fft2, s, T, hop, false) : torch_istft(*fftb, s, T, hop, false);
    }

    // wave [2][chunk] -> [stems][2][chunk]
    void run_chunk(const Graph& g, const float* wave, float* out) const {
        const int L = chunk(), Fn = n_fft / 2 + 1;
        const size_t plane = (size_t)dim_f * dim_t;
        std::vector<float> in(4 * plane), y((size_t)4 * n_stems * plane);
        for (int c = 0; c < 2; c++) {
            int T = 0;
            const std::vector<cplx> s = stft(wave + (size_t)c * L, T);
            if (T != dim_t) throw std::runtime_error("tfc_tdf: STFT frame count mismatch");
            for (int f = zero_bins; f < dim_f; f++)    // UVR zeroes the lowest bins of its input
                for (int t = 0; t < T; t++) {
                    const cplx v = s[(size_t)f * T + t];
                    in[(2 * c) * plane + (size_t)f * T + t] = (float)v.real();
                    in[(2 * c + 1) * plane + (size_t)f * T + t] = (float)v.imag();
                }
        }
        ggml_backend_tensor_set(g.input, in.data(), 0, ggml_nbytes(g.input));
        std::string error;
        if (!graph_compute_checked(backend, g.gf, "tfc_tdf", error)) throw std::runtime_error(error);
        ggml_backend_tensor_get(g.output, y.data(), 0, ggml_nbytes(g.output));
        std::vector<cplx> s((size_t)Fn * dim_t);
        for (int k = 0; k < n_stems; k++)
            for (int c = 0; c < 2; c++) {
                const float* re = &y[(size_t)(4 * k + 2 * c) * plane];
                const float* im = re + plane;
                std::fill(s.begin(), s.end(), cplx(0.0, 0.0));            // bins above dim_f stay zero
                for (size_t i = 0; i < plane; i++) s[i] = cplx(re[i], im[i]);
                const std::vector<float> w = istft(s, dim_t);
                std::copy(w.begin(), w.end(), out + ((size_t)k * 2 + c) * L);
            }
    }

    // ---- chunking -------------------------------------------------------------------------

    // UVR SeperateMDX.demix (separate.py): `trim` zeros in front, chunks of hop * (dim_t - 1)
    // every chunk - n_fft by default, each weighted by np.hanning of its unpadded length, the
    // trimmed edges dropped. Returns the model's stems [S][2][len], before compensation.
    std::vector<float> demix_uvr(const Graph& g, const float* mix, int len, float overlap,
                                 const std::function<void(int, int)>& tick) const {
        const int C = chunk(), trim = n_fft / 2, gen = C - 2 * trim;
        if (gen <= 0) throw std::runtime_error("tfc_tdf: chunk shorter than the FFT");
        const int64_t M = (int64_t)trim + len + gen + trim - len % gen;
        const int64_t step = overlap < 0 ? C - n_fft : (int64_t)((1.0 - (double)overlap) * C);
        if (step <= 0) throw std::invalid_argument("overlap too large for this model");
        std::vector<float> x((size_t)2 * M, 0.0f);
        for (int c = 0; c < 2; c++) std::copy(mix + (size_t)c * len, mix + (size_t)(c + 1) * len, &x[(size_t)c * M + trim]);
        const int S = n_stems, total = (int)((M + step - 1) / step);
        std::vector<double> result((size_t)S * 2 * M, 0.0), divider(M, 0.0);
        std::vector<float> part((size_t)2 * C), y((size_t)S * 2 * C);
        int done = 0;
        for (int64_t i = 0; i < M; i += step) {
            const int actual = (int)std::min<int64_t>(C, M - i);
            for (int c = 0; c < 2; c++) {
                std::fill(&part[(size_t)c * C], &part[(size_t)(c + 1) * C], 0.0f);
                std::copy_n(&x[(size_t)c * M + i], actual, &part[(size_t)c * C]);
            }
            tick(0, total);                    // cancellation only
            run_chunk(g, part.data(), y.data());
            const std::vector<double> w = overlap == 0 ? std::vector<double>(actual, 1.0) : hanning(actual);
            for (int k = 0; k < S * 2; k++)
                for (int t = 0; t < actual; t++) result[(size_t)k * M + i + t] += (double)y[(size_t)k * C + t] * w[t];
            for (int t = 0; t < actual; t++) divider[i + t] += w[t];
            tick(++done, total);
        }
        std::vector<float> out((size_t)S * 2 * len);
        for (int k = 0; k < S * 2; k++)
            for (int t = 0; t < len; t++) out[(size_t)k * len + t] = (float)(result[(size_t)k * M + trim + t] / divider[trim + t]);
        return out;
    }

    // ZFTurbo's utils.model_utils.demix, generic mode with batch_size 1: reflect-padded borders,
    // chunks every chunk / num_overlap, linear fades except at the first chunk's start and the
    // last chunk's end, short final chunks reflect-padded when over half a chunk long.
    std::vector<float> demix_msst(const Graph& g, const float* mix, int len, float overlap,
                                  const std::function<void(int, int)>& tick) const {
        const int C = chunk(), S = n_stems, fade = C / 10;
        const int step = overlap < 0 ? C / num_overlap : std::max(1, (int)std::lround(C * (1.0 - overlap)));
        const int border = C - step;
        const bool pad = len > 2 * border && border > 0;
        const int Lp = pad ? len + 2 * border : len;
        std::vector<float> x((size_t)2 * Lp);
        for (int c = 0; c < 2; c++) {
            const float* src = mix + (size_t)c * len;
            if (pad) {
                const std::vector<double> p = reflect_pad(src, len, border, border);
                std::transform(p.begin(), p.end(), &x[(size_t)c * Lp], [](double v) { return (float)v; });
            } else {
                std::copy(src, src + len, &x[(size_t)c * Lp]);
            }
        }
        std::vector<float> window(C, 1.0f);
        if (fade > 0) {
            const std::vector<float> in = linspace(0, 1, fade), out = linspace(1, 0, fade);
            std::copy(out.begin(), out.end(), window.end() - fade);
            std::copy(in.begin(), in.end(), window.begin());
        }
        const int total = (Lp + step - 1) / step;
        std::vector<double> result((size_t)S * 2 * Lp, 0.0), counter(Lp, 0.0);
        std::vector<float> part((size_t)2 * C), y((size_t)S * 2 * C);
        int done = 0;
        for (int i = 0; i < Lp;) {
            const int start = i, n = std::min(C, Lp - i);
            for (int c = 0; c < 2; c++) {
                const float* src = &x[(size_t)c * Lp + i];
                float* dst = &part[(size_t)c * C];
                std::copy(src, src + n, dst);
                if (n < C) {
                    if (n > C / 2) {
                        const std::vector<double> p = reflect_pad(src, n, 0, C - n);
                        std::transform(p.begin(), p.end(), dst, [](double v) { return (float)v; });
                    } else {
                        std::fill(dst + n, dst + C, 0.0f);
                    }
                }
            }
            i += step;
            tick(0, total);                    // cancellation only
            run_chunk(g, part.data(), y.data());
            std::vector<float> w = window;
            if (i - step == 0) std::fill(w.begin(), w.begin() + fade, 1.0f);
            else if (i >= Lp) std::fill(w.end() - fade, w.end(), 1.0f);
            for (int k = 0; k < S * 2; k++)
                for (int t = 0; t < n; t++) result[(size_t)k * Lp + start + t] += (double)y[(size_t)k * C + t] * w[t];
            for (int t = 0; t < n; t++) counter[start + t] += w[t];
            tick(++done, total);
        }
        std::vector<float> est((size_t)S * 2 * len);
        const int off = pad ? border : 0;
        for (int k = 0; k < S * 2; k++)
            for (int t = 0; t < len; t++) {
                const double v = result[(size_t)k * Lp + off + t] / counter[off + t];
                est[(size_t)k * len + t] = std::isfinite(v) ? (float)v : 0.0f;    // np.nan_to_num
            }
        return est;
    }
};

TFCTDF::TFCTDF(const std::string& path, const char* device, int cpu_threads) : p_(std::make_unique<Impl>()) {
    Impl& P = *p_;
    P.backend = make_backend(cpu_threads, device);
    P.model = load_gguf(path.c_str(), P.backend);
    const GgufModel& m = P.model;
    gguf_context* gg = m.gguf;
    if (m.string("general.architecture") != P.arch || m.u32("tfc_tdf.schema_version") != 1)
        throw std::runtime_error(path + ": unsupported tfc_tdf architecture/schema");
    P.name = m.string("stems.model");
    P.variant = m.string("tfc_tdf.variant");
    P.chunking = m.string("tfc_tdf.chunking");
    const int ks = gguf_find_key(gg, "stems.sources");
    if (ks < 0 || gguf_get_kv_type(gg, ks) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gg, ks) != GGUF_TYPE_STRING)
        throw std::runtime_error(path + ": missing stems.sources");
    for (size_t i = 0; i < gguf_get_arr_n(gg, ks); i++) P.sources.push_back(gguf_get_arr_str(gg, ks, i));
    P.n_stems = (int)P.sources.size();
    if (gguf_find_key(gg, "stems.complement") >= 0) {
        P.complement = true;
        P.sources.push_back(m.string("stems.complement"));
    }
    P.sr = (int)m.u32("stems.samplerate");
    P.n_fft = (int)m.u32("tfc_tdf.n_fft");
    P.hop = (int)m.u32("tfc_tdf.hop");
    P.dim_f = (int)m.u32("tfc_tdf.dim_f");
    P.dim_t = (int)m.u32("tfc_tdf.dim_t");
    P.subbands = (int)m.u32("tfc_tdf.num_subbands");
    P.scales = (int)m.u32("tfc_tdf.num_scales");
    P.blocks = (int)m.u32("tfc_tdf.num_blocks");
    P.channels = (int)m.u32("tfc_tdf.num_channels");
    P.growth = (int)m.u32("tfc_tdf.growth");
    P.bottleneck = (int)m.u32("tfc_tdf.bottleneck");
    P.compensate = m.f32("tfc_tdf.compensate");
    const std::string act = m.string("tfc_tdf.act");
    P.gelu = act == "gelu";
    if (P.chunking == "uvr_mdx") P.zero_bins = (int)m.u32("tfc_tdf.zero_bins");
    else if (P.chunking == "msst") P.num_overlap = (int)m.u32("tfc_tdf.num_overlap");
    else throw std::runtime_error(path + ": unknown tfc_tdf.chunking '" + P.chunking + "'");

    const int down = P.subbands << P.scales;
    bool ok = (P.variant == "mdx_net" || P.variant == "mdx23c") && (act == "relu" || act == "gelu") &&
              (P.variant != "mdx_net" || (act == "relu" && P.subbands == 1)) &&
              m.u32("stems.audio_channels") == 2 && P.sr > 0 && P.n_stems >= 1 && P.n_stems <= 16 &&
              P.n_fft >= 16 && P.n_fft <= 32768 && P.n_fft % 2 == 0 && P.hop > 0 && P.hop <= P.n_fft &&
              P.dim_f > 0 && P.dim_f <= P.n_fft / 2 + 1 && P.dim_f % down == 0 &&
              P.dim_t >= 2 && P.dim_t % (1 << P.scales) == 0 && P.scales >= 1 && P.scales <= 8 &&
              P.subbands >= 1 && P.blocks >= 1 && P.blocks <= 16 && P.channels > 0 && P.growth > 0 &&
              P.channels <= 4096 && P.growth <= 4096 && P.bottleneck >= 1 &&
              std::isfinite(P.compensate) && P.compensate > 0 && P.zero_bins >= 0 && P.zero_bins <= P.dim_f &&
              (P.chunking != "msst" || (P.num_overlap >= 1 && P.num_overlap <= P.chunk())) &&
              P.chunk() > P.n_fft / 2;
    for (int i = 0, f = P.dim_f / P.subbands; ok && i <= P.scales; i++, f /= 2) ok = f % P.bottleneck == 0;
    if (!ok) throw std::runtime_error(path + ": invalid tfc_tdf configuration");
    P.validate_weights();
    if ((P.n_fft & (P.n_fft - 1)) == 0) P.fft2 = std::make_unique<FFT>(P.n_fft);
    else P.fftb = std::make_unique<VRFFT>(P.n_fft);
}

TFCTDF::~TFCTDF() = default;
const std::string& TFCTDF::name() const { return p_->name; }
const std::string& TFCTDF::architecture() const { return p_->arch; }
const std::vector<std::string>& TFCTDF::sources() const { return p_->sources; }
int TFCTDF::samplerate() const { return p_->sr; }
int TFCTDF::segment_samples() const { return p_->chunk(); }
const char* TFCTDF::backend_name() const { return ggml_backend_name(p_->backend); }

std::vector<float> TFCTDF::forward(const float* chunk) const {
    auto g = p_->build();
    std::vector<float> out((size_t)p_->n_stems * 2 * p_->chunk());
    p_->run_chunk(*g, chunk, out.data());
    return out;
}

std::vector<float> TFCTDF::separate(const float* mix, int len, const SeparateOptions& opt) const {
    const Impl& P = *p_;
    if (!mix || len <= 0) throw std::runtime_error("tfc_tdf: empty input");
    for (size_t k = 0; k < (size_t)2 * len; k++)
        if (!std::isfinite(mix[k])) throw std::runtime_error("tfc_tdf: non-finite input");
    if (opt.shifts != 0) throw std::runtime_error("tfc_tdf: shifts are unsupported");
    if (opt.overlap >= 1.0f) throw std::invalid_argument("overlap must be < 1");
    auto tick = [&](int done, int total) {
        if (opt.should_cancel && opt.should_cancel()) throw Cancelled();
        if (opt.progress && done > 0) opt.progress(done, total);
    };
    if (opt.should_cancel && opt.should_cancel()) throw Cancelled();
    auto g = P.build();
    std::vector<float> est = P.chunking == "uvr_mdx" ? P.demix_uvr(*g, mix, len, opt.overlap, tick)
                                                     : P.demix_msst(*g, mix, len, opt.overlap, tick);
    const size_t n = (size_t)2 * len;
    if (P.compensate != 1.0f)
        for (float& v : est) v *= P.compensate;
    if (P.complement) {                // UVR's secondary stem: the mix minus the compensated stems
        est.resize(est.size() + n);
        float* rest = &est[(size_t)P.n_stems * n];
        for (size_t i = 0; i < n; i++) {
            float v = mix[i];
            for (int k = 0; k < P.n_stems; k++) v -= est[(size_t)k * n + i];
            rest[i] = v;
        }
    }
    return est;
}

} // namespace st

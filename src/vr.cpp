#include "vr.h"
#include "vr_dsp.h"
#include "gguf_model.h"
#include "ggml-alloc.h"
#include <cstring>

namespace st {
namespace {
struct VRGraph {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor *input = nullptr, *output = nullptr;
    std::vector<ggml_tensor*> zeros;
    ~VRGraph() { if (alloc) ggml_gallocr_free(alloc); if (ctx) ggml_free(ctx); }
};
ggml_tensor* cat(ggml_context* ctx, std::vector<ggml_tensor*> v, int dim) {
    while (v.size() > 1) {
        std::vector<ggml_tensor*> next;
        for (size_t i = 0; i+1 < v.size(); i += 2) next.push_back(ggml_concat(ctx, v[i], v[i+1], dim));
        if (v.size()%2) next.push_back(v.back());
        v.swap(next);
    }
    return v.at(0);
}
}

struct VR::Impl {
    ggml_backend_t backend = nullptr;
    GgufModel model;
    std::string name, arch = "vr_cascaded";
    std::vector<std::string> sources;
    int sr = 0, bins = 0, nout = 0, lstm = 0, window = 0, offset = 64, pre_start = 0, pre_stop = 0;
    std::vector<VRBand> bands;
    std::map<std::pair<int,int>, std::vector<float>> firs;
    ~Impl() { model.free(); if (backend) ggml_backend_free(backend); }
    ggml_tensor* W(const std::string& n) const { return model.get(n); }

    void validate_weights() const {
        auto shape = [&](const std::string& n, int64_t a, int64_t b=1, int64_t c=1, int64_t d=1) {
            auto t = W(n);
            if (t->type != GGML_TYPE_F32 || t->ne[0]!=a || t->ne[1]!=b || t->ne[2]!=c || t->ne[3]!=d)
                throw std::runtime_error("vr: invalid F32 tensor shape: "+n);
        };
        auto cb_shape = [&](const std::string& n, int in, int out, int kernel=3) {
            shape(n+".conv.weight",kernel,kernel,in,out);
            shape(n+".conv.bias",1,1,out);
        };
        auto base_shape = [&](const std::string& p, int in, int out, int freq, int nl) {
            cb_shape(p+".enc1",in,out);
            const int scale[] = {1,2,4,6,8};
            for (int i=1; i<5; i++) {
                const auto n=p+".enc"+std::to_string(i+1);
                cb_shape(n+".conv1",out*scale[i-1],out*scale[i]);
                cb_shape(n+".conv2",out*scale[i],out*scale[i]);
            }
            cb_shape(p+".aspp.conv1.1",out*8,out*8,1);
            cb_shape(p+".aspp.conv2",out*8,out*8,1);
            for (int i=3; i<=5; i++) cb_shape(p+".aspp.conv"+std::to_string(i),out*8,out*8);
            cb_shape(p+".aspp.bottleneck",out*40,out*8,1);
            cb_shape(p+".dec4.conv1",out*14,out*6);
            cb_shape(p+".dec3.conv1",out*10,out*4);
            cb_shape(p+".dec2.conv1",out*6,out*2);
            cb_shape(p+".dec1.conv1",out*3+1,out);
            cb_shape(p+".lstm_dec2.conv",out*2,1,1);
            for (const std::string s : {std::string(),std::string("_reverse")}) {
                shape(p+".lstm_dec2.lstm.weight_ih_l0"+s,freq/2,nl*2);
                shape(p+".lstm_dec2.lstm.weight_hh_l0"+s,nl/2,nl*2);
                shape(p+".lstm_dec2.lstm.bias_ih_l0"+s,nl*2);
                shape(p+".lstm_dec2.lstm.bias_hh_l0"+s,nl*2);
            }
            shape(p+".lstm_dec2.dense.weight",nl,freq/2);
            shape(p+".lstm_dec2.dense.bias",freq/2);
        };
        base_shape("stg1_low_band_net.0",2,nout/2,bins/2,lstm);
        cb_shape("stg1_low_band_net.1",nout/2,nout/4,1);
        base_shape("stg1_high_band_net",2,nout/4,bins/2,lstm/2);
        base_shape("stg2_low_band_net.0",nout/4+2,nout,bins/2,lstm);
        cb_shape("stg2_low_band_net.1",nout,nout/2,1);
        base_shape("stg2_high_band_net",nout/4+2,nout/2,bins/2,lstm/2);
        base_shape("stg3_full_band_net",nout*3/4+2,nout,bins,lstm);
        shape("out.weight",1,1,nout,2);
    }

    ggml_tensor* mm(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x) const {
        auto y = ggml_mul_mat(ctx, w, x);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        return y;
    }
    ggml_tensor* conv(ggml_context* ctx, ggml_tensor* x, const std::string& n,
                      int stride=1, int pad=0, int d0=1, int d1=1, bool bias=true) const {
        auto k = W(n+".weight");
        if (k->ne[2] != x->ne[2]) throw std::runtime_error("vr: convolution input mismatch: "+n);
        auto im = ggml_im2col(ctx, k, x, stride, stride, pad*d0, pad*d1, d0, d1, true, GGML_TYPE_F32);
        auto y = mm(ctx, ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1]*im->ne[2]),
                         ggml_reshape_2d(ctx, k, k->ne[0]*k->ne[1]*k->ne[2], k->ne[3]));
        y = ggml_reshape_3d(ctx, y, im->ne[1], im->ne[2], k->ne[3]);
        return bias ? ggml_add(ctx, y, W(n+".bias")) : y;
    }
    ggml_tensor* cb(ggml_context* ctx, ggml_tensor* x, const std::string& n,
                    int stride=1, int pad=0, bool leaky=false, int d0=1, int d1=1) const {
        auto y = conv(ctx, x, n+".conv", stride, pad, d0, d1);
        return leaky ? ggml_leaky_relu(ctx, y, 0.01f, false) : ggml_relu(ctx, y);
    }
    ggml_tensor* up(ggml_context* ctx, ggml_tensor* x, int t, int f) const {
        return ggml_interpolate(ctx, x, t, f, x->ne[2], 1,
                                GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
    }
    ggml_tensor* decoder(ggml_context* ctx, ggml_tensor* x, ggml_tensor* skip, const std::string& n) const {
        x = up(ctx, x, (int)x->ne[0]*2, (int)x->ne[1]*2);
        if (skip->ne[0] != x->ne[0] || skip->ne[1] != x->ne[1]) throw std::runtime_error("vr: incompatible decoder shape");
        return cb(ctx, ggml_concat(ctx, x, skip, 2), n+".conv1", 1, 1);
    }
    ggml_tensor* recurrent(VRGraph& g, ggml_tensor* x, const std::string& n) const {
        auto ctx = g.ctx;
        x = cb(ctx, x, n+".conv");                       // [T,F,1]
        const int T = (int)x->ne[0], F = (int)x->ne[1];
        x = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, x, T, F))); // [F,T]
        std::vector<ggml_tensor*> directions;
        for (int rev = 0; rev < 2; rev++) {
            const std::string s = rev ? "_reverse" : "";
            auto wi = W(n+".lstm.weight_ih_l0"+s), wh = W(n+".lstm.weight_hh_l0"+s);
            const int H = (int)wh->ne[0];
            if (wi->ne[0] != F || wi->ne[1] != 4*H || wh->ne[1] != 4*H) throw std::runtime_error("vr: invalid LSTM weights");
            auto gates = ggml_add(ctx, mm(ctx, wi, x),
                ggml_add(ctx, W(n+".lstm.bias_ih_l0"+s), W(n+".lstm.bias_hh_l0"+s)));
            auto h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
            auto c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
            ggml_set_input(h); ggml_set_input(c);
            g.zeros.push_back(h); g.zeros.push_back(c);
            std::vector<ggml_tensor*> frames(T);
            for (int j = 0; j < T; j++) {
                const int t = rev ? T-1-j : j;
                auto v = ggml_add(ctx, ggml_view_2d(ctx, gates, 4*H, 1, gates->nb[1], (size_t)t*gates->nb[1]), mm(ctx, wh, h));
                auto part = [&](int k) { return ggml_cont(ctx, ggml_view_2d(ctx, v, H, 1, v->nb[1], (size_t)k*H*sizeof(float))); };
                auto i = ggml_sigmoid(ctx, part(0)), f = ggml_sigmoid(ctx, part(1));
                auto z = ggml_tanh(ctx, part(2)), o = ggml_sigmoid(ctx, part(3));
                c = ggml_add(ctx, ggml_mul(ctx, f, c), ggml_mul(ctx, i, z));
                h = ggml_mul(ctx, o, ggml_tanh(ctx, c));
                frames[t] = h;
            }
            directions.push_back(cat(ctx, frames, 1));
        }
        auto y = ggml_concat(ctx, directions[0], directions[1], 0);  // [2H,T]
        y = ggml_relu(ctx, ggml_add(ctx, mm(ctx, W(n+".dense.weight"), y), W(n+".dense.bias")));
        return ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, y)), T, F, 1);
    }
    ggml_tensor* base(VRGraph& g, ggml_tensor* x, const std::string& n) const {
        auto ctx = g.ctx;
        std::vector<ggml_tensor*> e;
        x = cb(ctx, x, n+".enc1", 1, 1); e.push_back(x);
        for (int k = 2; k <= 5; k++) {
            const std::string p = n+".enc"+std::to_string(k);
            x = cb(ctx, x, p+".conv1", 2, 1, true);
            x = cb(ctx, x, p+".conv2", 1, 1, true);
            e.push_back(x);
        }
        // AdaptiveAvgPool2d((1,None)): average frequency independently for each time.
        auto pool = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)); // [F,T,C]
        pool = ggml_mean(ctx, pool);
        pool = ggml_cont(ctx, ggml_permute(ctx, pool, 1, 0, 2, 3));
        std::vector<ggml_tensor*> a = {up(ctx, cb(ctx, pool, n+".aspp.conv1.1"), (int)x->ne[0], (int)x->ne[1]),
                                      cb(ctx, x, n+".aspp.conv2")};
        for (int k = 0; k < 3; k++) a.push_back(cb(ctx, x, n+".aspp.conv"+std::to_string(k+3), 1, 1, false, 2*(k+1), 4*(k+1)));
        x = cb(ctx, cat(ctx, a, 2), n+".aspp.bottleneck");
        for (int k = 4; k >= 2; k--) x = decoder(ctx, x, e[k-1], n+".dec"+std::to_string(k));
        x = ggml_concat(ctx, x, recurrent(g, x, n+".lstm_dec2"), 2);
        return decoder(ctx, x, e[0], n+".dec1");
    }
    std::unique_ptr<VRGraph> build() const {
        auto g = std::make_unique<VRGraph>();
        const size_t nodes = (size_t)window*256; // recurrent graph scales with frame count
        ggml_init_params init = {nodes*ggml_tensor_overhead()+ggml_graph_overhead_custom(nodes, false), nullptr, true};
        g->ctx = ggml_init(init);
        if (!g->ctx) throw std::runtime_error("vr: graph context allocation failed");
        auto ctx = g->ctx;
        g->input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, window, bins+1, 2);
        ggml_set_input(g->input);
        auto x = ggml_view_3d(ctx, g->input, window, bins, 2, g->input->nb[1], g->input->nb[2], 0);
        auto low = ggml_cont(ctx, ggml_view_3d(ctx, x, window, bins/2, 2, x->nb[1], x->nb[2], 0));
        auto high = ggml_cont(ctx, ggml_view_3d(ctx, x, window, bins/2, 2, x->nb[1], x->nb[2], (size_t)(bins/2)*x->nb[1]));
        auto l1 = cb(ctx, base(*g, low, "stg1_low_band_net.0"), "stg1_low_band_net.1");
        auto h1 = base(*g, high, "stg1_high_band_net");
        auto l2 = cb(ctx, base(*g, ggml_concat(ctx, low, l1, 2), "stg2_low_band_net.0"), "stg2_low_band_net.1");
        auto h2 = base(*g, ggml_concat(ctx, high, h1, 2), "stg2_high_band_net");
        x = cat(ctx, {ggml_cont(ctx, x), ggml_concat(ctx, l1, h1, 1), ggml_concat(ctx, l2, h2, 1)}, 2);
        x = ggml_sigmoid(ctx, conv(ctx, base(*g, x, "stg3_full_band_net"), "out", 1, 0, 1, 1, false));
        // Replicate the highest estimated bin to the Nyquist/extra bin.
        auto last = ggml_cont(ctx, ggml_view_3d(ctx, x, window, 1, 2, x->nb[1], x->nb[2], (size_t)(bins-1)*x->nb[1]));
        x = ggml_concat(ctx, x, last, 1);
        g->output = ggml_cont(ctx, ggml_view_3d(ctx, x, window-2*offset, bins+1, 2, x->nb[1], x->nb[2], offset*sizeof(float)));
        ggml_set_output(g->output);
        g->gf = ggml_new_graph_custom(ctx, nodes, false);
        ggml_build_forward_expand(g->gf, g->output);
        g->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(g->alloc, g->gf)) throw std::runtime_error("vr: graph allocation failed");
        return g;
    }
    std::vector<float> run(const VRGraph& g, const float* mag) const {
        ggml_backend_tensor_set(g.input, mag, 0, ggml_nbytes(g.input));
        for (auto t : g.zeros) {
            const std::vector<float> zero((size_t)ggml_nelements(t), 0);
            ggml_backend_tensor_set(t, zero.data(), 0, ggml_nbytes(t));
        }
        std::string error;
        if (!graph_compute_checked(backend, g.gf, "vr", error)) throw std::runtime_error(error);
        std::vector<float> y((size_t)ggml_nelements(g.output));
        ggml_backend_tensor_get(g.output, y.data(), 0, y.size()*sizeof(float));
        return y;
    }
    std::vector<float> resample(const std::vector<float>& x, int n, int src, int dst, int& out_n) const {
        if (src == dst) { out_n = n; return x; }
        return vr_resample(x, n, src, dst, firs.at({src,dst}), out_n);
    }
    std::vector<cplx> spectrum(const float* mix, int len, int& T) const {
        std::vector<std::vector<cplx>> specs(bands.size());
        std::vector<int> frames(bands.size());
        std::vector<float> wave(mix, mix+(size_t)2*len);
        int n = len, rate = sr;
        T = INT32_MAX;
        for (int i = (int)bands.size()-1; i >= 0; i--) {
            wave = resample(wave, n, rate, bands[i].sr, n);
            rate = bands[i].sr;
            specs[i] = vr_stft(wave, n, bands[i], frames[i]);
            T = std::min(T, frames[i]);
        }
        std::vector<cplx> spec((size_t)2*(bins+1)*T);
        int off = 0;
        for (size_t i = 0; i < bands.size(); i++) {
            const auto& b = bands[i];
            const int F = b.nfft/2+1;
            for (int c = 0; c < 2; c++)
                for (int f = b.start; f < b.stop; f++)
                    for (int t = 0; t < T; t++) spec[((size_t)c*(bins+1)+off+f-b.start)*T+t] = specs[i][((size_t)c*F+f)*frames[i]+t];
            off += b.stop-b.start;
        }
        if (pre_start > 0)
            for (int c = 0; c < 2; c++)
                for (int f = 0; f <= bins; f++)
                    for (int t = 0; t < T; t++) spec[((size_t)c*(bins+1)+f)*T+t] *= vr_lowpass(f, pre_start, pre_stop);
        return spec;
    }
    std::vector<float> synthesize(const std::vector<cplx>& spec, int T, int len) const {
        std::vector<float> wave;
        int off = 0, n = 0;
        for (size_t i = 0; i < bands.size(); i++) {
            const auto& b = bands[i];
            const int F = b.nfft/2+1;
            std::vector<cplx> part((size_t)2*F*T);
            for (int c = 0; c < 2; c++)
                for (int f = b.start; f < b.stop; f++) {
                    const float gain = vr_highpass(f, b.hp_start, b.hp_stop-1) *
                        (i+1 == bands.size() ? 1 : vr_lowpass(f, b.lp_start, b.lp_stop));
                    for (int t = 0; t < T; t++) part[((size_t)c*F+f)*T+t] = spec[((size_t)c*(bins+1)+off+f-b.start)*T+t]*(double)gain;
                }
            off += b.stop-b.start;
            auto add = vr_istft(part, T, b);
            const int bn = b.hop*(T-1);
            if (wave.empty()) { wave = std::move(add); n = bn; }
            else {
                if (n != bn) throw std::runtime_error("vr: synthesis bands do not align");
                for (size_t k = 0; k < wave.size(); k++) wave[k] += add[k];
            }
            if (i+1 < bands.size()) wave = resample(wave, n, b.sr, bands[i+1].sr, n);
        }
        std::vector<float> out((size_t)2*len);
        for (int c = 0; c < 2; c++) std::copy_n(wave.data()+(size_t)c*n, std::min(n,len), out.data()+(size_t)c*len);
        return out;
    }
};

VR::VR(const std::string& path, const char* device, int cpu_threads) : p_(std::make_unique<Impl>()) {
    auto& P = *p_;
    P.backend = make_backend(cpu_threads, device);
    P.model = load_gguf(path.c_str(), P.backend);
    const auto& m = P.model;
    if (m.string("general.architecture") != P.arch || m.u32("vr.schema_version") != 1)
        throw std::runtime_error("vr: unsupported architecture/schema");
    P.name = m.string("stems.model");
    int k = gguf_find_key(m.gguf, "stems.sources");
    if (k < 0 || gguf_get_kv_type(m.gguf,k) != GGUF_TYPE_ARRAY || gguf_get_arr_type(m.gguf,k) != GGUF_TYPE_STRING || gguf_get_arr_n(m.gguf,k) != 2)
        throw std::runtime_error("vr: expected primary and secondary sources");
    for (int i = 0; i < 2; i++) P.sources.push_back(gguf_get_arr_str(m.gguf,k,i));
    P.sr = (int)m.u32("stems.samplerate");
    P.bins = (int)m.u32("vr.bins"); P.nout = (int)m.u32("vr.nout"); P.lstm = (int)m.u32("vr.nout_lstm");
    P.window = (int)m.u32("vr.window"); P.offset = (int)m.u32("vr.offset");
    P.pre_start = (int)m.u32("vr.pre_filter_start"); P.pre_stop = (int)m.u32("vr.pre_filter_stop");
    if (m.u32("stems.audio_channels") != 2 || P.sr <= 0 || P.bins < 32 || P.bins > 8192 || P.bins%32 ||
        P.nout < 4 || P.nout > 256 || P.nout%4 || P.lstm < 4 || P.lstm > 512 || P.lstm%4 ||
        P.window < 144 || P.window > 1024 || P.window%16 || P.offset != 64 || P.pre_start < 0 || P.pre_stop > P.bins || P.pre_stop <= P.pre_start ||
        m.string("vr.resampling") != "polyphase") throw std::runtime_error("vr: invalid configuration");
    const int count = (int)m.u32("vr.band_count");
    if (count < 1 || count > 16) throw std::runtime_error("vr: invalid band count");
    int total = 0;
    for (int i = 0; i < count; i++) {
        auto get = [&](const char* key) {
            const auto n = "vr.band."+std::to_string(i)+"."+key;
            int idx = gguf_find_key(m.gguf,n.c_str());
            if (idx < 0 || gguf_get_kv_type(m.gguf,idx) != GGUF_TYPE_INT32) throw std::runtime_error("vr: missing band configuration");
            return gguf_get_val_i32(m.gguf,idx);
        };
        VRBand b = {get("sr"), get("hl"), get("n_fft"), get("crop_start"), get("crop_stop"),
                    get("hpf_start"), get("hpf_stop"), get("lpf_start"), get("lpf_stop")};
        if (b.sr <= 0 || b.hop <= 0 || b.nfft < 2 || b.nfft > 16384 || b.nfft%2 || b.start < 0 || b.stop <= b.start || b.stop > b.nfft/2+1)
            throw std::runtime_error("vr: invalid frequency band");
        if ((b.hp_start >= 0 && (b.hp_start < 1 || b.hp_stop < 0 || b.hp_stop >= b.hp_start || b.hp_start > b.nfft/2)) ||
            (b.lp_start >= 0 && (b.lp_start < 1 || b.lp_stop <= b.lp_start || b.lp_stop > b.nfft/2)))
            throw std::runtime_error("vr: invalid band filter");
        total += b.stop-b.start;
        P.bands.push_back(b);
    }
    if (total > P.bins || P.bands.back().sr != P.sr) throw std::runtime_error("vr: band layout mismatch");
    for (const auto& b : P.bands)
        if ((int64_t)b.sr*P.bands.back().hop != (int64_t)P.sr*b.hop) throw std::runtime_error("vr: frame duration mismatch");
    P.validate_weights();
    for (const auto& a : P.bands)
        for (const auto& b : P.bands)
            if (a.sr != b.sr && !P.firs.count({a.sr,b.sr})) {
                auto t = m.get("resample."+std::to_string(a.sr)+"."+std::to_string(b.sr));
                const int ratio = std::max(a.sr,b.sr)/std::gcd(a.sr,b.sr);
                const int64_t expected = (int64_t)20*ratio+1;
                if (t->type != GGML_TYPE_F32 || t->ne[1]!=1 || t->ne[2]!=1 || t->ne[3]!=1 ||
                    t->ne[0] != expected) throw std::runtime_error("vr: invalid resampling FIR");
                std::vector<float> h((size_t)expected);
                ggml_backend_tensor_get(t,h.data(),0,h.size()*sizeof(float));
                P.firs[{a.sr,b.sr}] = std::move(h);
            }
}
VR::~VR() = default;
const std::string& VR::name() const { return p_->name; }
const std::string& VR::architecture() const { return p_->arch; }
const std::vector<std::string>& VR::sources() const { return p_->sources; }
int VR::samplerate() const { return p_->sr; }
int VR::segment_samples() const { return p_->window*p_->bands.back().hop; }
const char* VR::backend_name() const { return ggml_backend_name(p_->backend); }
std::vector<float> VR::predict_mask(const float* mag) const { auto g = p_->build(); return p_->run(*g, mag); }

std::vector<float> VR::separate(const float* mix, int len, const SeparateOptions& opt) const {
    const auto& P = *p_;
    if (!mix || len <= 0) throw std::runtime_error("vr: empty input");
    for (size_t k = 0; k < (size_t)2*len; k++)
        if (!std::isfinite(mix[k])) throw std::runtime_error("vr: non-finite input");
    if (opt.shifts != 0 || opt.overlap >= 0) throw std::runtime_error("vr: shifts/overlap are unsupported; the model uses cropped spectral windows");
    auto cancel = [&] { if (opt.should_cancel && opt.should_cancel()) throw Cancelled(); };
    cancel();
    int T = 0;
    auto spec = P.spectrum(mix,len,T);
    const int F = P.bins+1, roi = P.window-2*P.offset, patches = T/roi+1;
    std::vector<float> mag(spec.size());
    float peak = 0;
    for (size_t k = 0; k < mag.size(); k++) { mag[k] = (float)std::abs(spec[k]); peak = std::max(peak, mag[k]); }
    if (!std::isfinite(peak)) throw std::runtime_error("vr: non-finite input");
    if (peak == 0) return std::vector<float>((size_t)4*len,0);
    auto g = P.build();
    std::vector<float> mask(spec.size()), patch((size_t)2*F*P.window);
    for (int i = 0; i < patches; i++) {
        cancel();
        std::fill(patch.begin(),patch.end(),0);
        for (int cf = 0; cf < 2*F; cf++)
            for (int t = 0; t < P.window; t++) {
                int src = i*roi+t-P.offset;
                if (src >= 0 && src < T) patch[(size_t)cf*P.window+t] = mag[(size_t)cf*T+src]/peak;
            }
        auto y = P.run(*g,patch.data());
        for (int cf = 0; cf < 2*F; cf++)
            for (int t = 0; t < roi && i*roi+t < T; t++) mask[(size_t)cf*T+i*roi+t] = y[(size_t)cf*roi+t];
        if (opt.progress) opt.progress(i+1,patches);
    }
    std::vector<float> out;
    out.reserve((size_t)4*len);
    for (int s = 0; s < 2; s++) {
        cancel();
        auto z = spec;
        for (size_t k = 0; k < z.size(); k++) z[k] *= s == 0 ? mask[k] : 1-mask[k];
        auto wave = P.synthesize(z,T,len);
        out.insert(out.end(),wave.begin(),wave.end());
    }
    return out;
}
} // namespace st

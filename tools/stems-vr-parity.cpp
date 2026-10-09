#include "vr.h"
#include "npy.h"
#include "wav.h"
#include "gguf_model.h"
#include <chrono>
#include <cmath>
#include <cstdio>

bool compare(const std::vector<float>& got, const std::vector<float>& ref, const char* label) {
    if (got.size() != ref.size()) throw std::runtime_error("parity shape mismatch");
    double dot=0, a=0, b=0, err=0;
    for (size_t i=0; i<got.size(); i++) {
        dot += (double)got[i]*ref[i]; a += (double)got[i]*got[i]; b += (double)ref[i]*ref[i];
        err += ((double)got[i]-ref[i])*((double)got[i]-ref[i]);
    }
    double cos=dot/std::sqrt(a*b+1e-30), snr=10*std::log10(b/(err+1e-30));
    printf("%s: cos %.9f, SNR %.2f dB\n",label,cos,snr);
    return std::isfinite(cos) && cos>=0.99999 && snr>=50;
}
int main(int argc, char** argv) {
    if (argc<4 || argc>5) { fprintf(stderr,"usage: stems-vr-parity MODEL.gguf REFS WAV [DEVICE]\n"); return 2; }
    try {
        st::VR m(argv[1],argc==5?argv[4]:nullptr,4);
        std::vector<size_t> shape;
        auto input=st::read_npy_f32(std::string(argv[2])+"/mask_in.npy",shape);
        const auto meta = st::load_gguf_metadata(argv[1]);
        if (shape != std::vector<size_t>{2,(size_t)meta.u32("vr.bins")+1,(size_t)meta.u32("vr.window")})
            throw std::runtime_error("invalid mask input shape");
        auto ref=st::read_npy_f32(std::string(argv[2])+"/mask_out.npy",shape);
        auto t=std::chrono::steady_clock::now();
        bool ok=compare(m.predict_mask(input.data()),ref,"mask");
        printf("mask forward %.2f s\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count());
        int n,ch,sr;
        auto audio=st::read_wav_planar(argv[3],n,ch,sr);
        if (ch!=2 || sr!=m.samplerate()) throw std::runtime_error("WAV must be stereo at the model rate");
        ref=st::read_npy_f32(std::string(argv[2])+"/full.npy",shape);
        t=std::chrono::steady_clock::now();
        auto full=m.separate(audio.data(),n);
        if (full.size()!=ref.size()) throw std::runtime_error("full reference shape mismatch");
        for (int s=0; s<2; s++) {
            const size_t a=(size_t)s*2*n, b=a+2*n;
            ok &= compare(std::vector<float>(full.begin()+a,full.begin()+b),
                          std::vector<float>(ref.begin()+a,ref.begin()+b),m.sources()[s].c_str());
        }
        printf("separate %.3f s audio: %.2f s\n",(double)n/sr,std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count());
        std::vector<float> silence(2*113,0);
        auto zero=m.separate(silence.data(),113);
        ok &= zero.size()==4*113;
        for (float v:zero) ok &= v==0;
        st::SeparateOptions opt; opt.should_cancel=[] {return true;};
        try { m.separate(audio.data(),n,opt); ok=false; } catch(const st::Cancelled&) {}
        puts(ok?"PASS":"FAIL");
        return ok?0:1;
    } catch (const std::exception& e) { fprintf(stderr,"error: %s\n",e.what()); return 1; }
}

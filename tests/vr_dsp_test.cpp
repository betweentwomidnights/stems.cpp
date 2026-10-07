#include "vr_dsp.h"
#include <cstdio>
#include <stdexcept>

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int main() {
    try {
        for (int n : {320,512,640,960,2048}) {
            st::VRFFT fft(n);
            std::vector<st::cplx> x(n);
            x[7]=1;
            fft.run(x,false);
            for (int k=0; k<n; k++) require(std::abs(x[k]-std::polar(1.0,-2*M_PI*7*k/n))<1e-10,"non-radix-2 FFT impulse");
            fft.run(x,true);
            for (int k=0; k<n; k++) require(std::abs(x[k]/(double)n-st::cplx(k==7?1:0,0))<1e-10,"FFT inverse");
            st::VRBand b={44100,n/2,n,0,n/2,-1,-1,-1,-1};
            int samples=n*3;
            std::vector<float> wave(2*samples);
            for (int k=0; k<2*samples; k++) wave[k]=(float)(.2*std::sin(.017*k));
            int frames=0;
            auto spec=st::vr_stft(wave,samples,b,frames);
            auto got=st::vr_istft(spec,frames,b);
            require(got.size()==wave.size(),"VR STFT length");
            for (size_t k=0; k<wave.size(); k++) require(std::abs(wave[k]-got[k])<1e-6,"VR STFT round trip");
        }
        require(st::vr_lowpass(24,25,53)==1 && st::vr_lowpass(52,25,53)==0,"VR lowpass edges");
        require(st::vr_highpass(85,130,85)==0 && st::vr_highpass(86,130,85)==0 && st::vr_highpass(131,130,85)==1,"VR highpass edges");
        // A symmetric 3-tap interpolation filter has a known half-sample response.
        std::vector<float> x={0,1,0,0,0,1,0,0};
        std::vector<float> h={.5f,1,.5f};
        int n=0;
        auto y=st::vr_resample(x,4,1,2,h,n);
        require(n==8 && y==std::vector<float>({0,.5f,1,.5f,0,0,0,0,0,.5f,1,.5f,0,0,0,0}),"centred polyphase alignment");
        puts("VR DSP PASS"); return 0;
    } catch (const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}

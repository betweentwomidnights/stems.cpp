---
license: other
license_name: unknown
pipeline_tag: audio-to-audio
tags:
- audio-source-separation
- denoising
- dereverberation
- uvr
- vr-arch
- gguf
- stems.cpp
---

# UVR VR denoise and de-echo models, GGUF

> **No license is stated for these weights anywhere upstream.** They are distributed with
> [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)'s model downloads
> and mirrored without one. They are republished here in GGUF form with that caveat. Do not assume
> any license, and if you hold the rights and want them removed, open a discussion on this repository.

Five of Ultimate Vocal Remover's VR-architecture (UVR 5.1 `CascadedNet`) models, converted to GGUF for
[stems.cpp](https://github.com/betweentwomidnights/stems.cpp), a C++/ggml stem separator that runs on
CPU, CUDA, Vulkan and Metal with no Python. Each returns two stems: the one the network's mask selects
(primary) and its complement.

| model id | file | outputs (primary, secondary) | size |
|---|---|---|---:|
| `uvr_denoise_lite` | `uvr_denoise_lite-4M-v1.0-F32.gguf` | noise, denoised | 16.8 MiB |
| `uvr_denoise` | `uvr_denoise-32M-v1.0-F32.gguf` | noise, denoised | 120.8 MiB |
| `uvr_deecho_normal` | `uvr_deecho_normal-32M-v1.0-F32.gguf` | no_echo, echo | 120.8 MiB |
| `uvr_deecho_aggressive` | `uvr_deecho_aggressive-32M-v1.0-F32.gguf` | echo, no_echo | 120.8 MiB |
| `uvr_deecho_dereverb` | `uvr_deecho_dereverb-56M-v1.0-F32.gguf` | no_reverb, reverb | 212.8 MiB |

F32 only. `SHA256SUMS` lists every file.

## Download and run

```sh
git clone --recursive https://github.com/betweentwomidnights/stems.cpp && cd stems.cpp
./build.sh vulkan                          # or cpu | cuda | metal   (Windows: build.cmd)
./models.sh uvr_denoise uvr_deecho_normal
build-vulkan/bin/stems-split -m models/uvr_denoise-32M-v1.0-F32.gguf -i recording.wav -o clean/ --stems denoised
build-vulkan/bin/stems-split -m models/uvr_deecho_normal-32M-v1.0-F32.gguf -i vocal.wav -o dry/ --stems no_echo
```

`stems-server` finds each by model id (`"model": "uvr_denoise"`). The stems.cpp release packages
(v0.1.3 and later) load them through the same C ABI as the other models.

## What the runtime does

The network is UVR's 5.1 `CascadedNet`, unchanged. The spectral path around it is defined explicitly
so every platform computes the same thing: periodic Hann STFT with centred zero padding, UVR's band
layout (`4band_v3` for the 32M/56M models, `1band_sr44100_hl1024` for Lite), polyphase resampling
between bands with the filters stored in the GGUF, UVR's prefilter and synthesis masks, and
256-frame windows with 64 frames discarded at each edge. It is **not** byte-identical to the UVR GUI:
UVR's Windows synthesis normally uses `sinc_fastest`, and its optional aggressiveness, test-time
augmentation, artifact merging and high-end mirroring are not implemented. See stems.cpp's
`docs/VR.md`.

## Parity

SNR of the C++ output against the unchanged UVR network and stems.cpp's portable Python pipeline, on a
4 s stereo music excerpt, measured 2026-10-07/09 with ggml `9d0d910b`. CPU: Core Ultra 9 275HX.
Vulkan and CUDA: RTX 5070 Laptop. Every run passes stems.cpp's check (cosine >= 0.99999, SNR >= 50 dB).

| model | backend | mask | primary | secondary |
|---|---|---:|---:|---:|
| DeNoise-Lite | CPU / Vulkan / CUDA | 90.8 / 104.9 / 106.3 dB | 110.3 / 112.0 / 114.2 dB | 138.9 / 139.2 / 139.4 dB |
| DeNoise | CPU / Vulkan / CUDA | 112.3 / 114.1 / 111.4 dB | 115.1 / 115.8 / 112.7 dB | 135.1 / 135.1 / 135.1 dB |
| De-Echo Normal | CPU / Vulkan / CUDA | 118.7 / 119.3 / 110.8 dB | 123.1 / 122.7 / 115.7 dB | 110.7 / 110.2 / 103.0 dB |
| De-Echo Aggressive | CPU / Vulkan / CUDA | 114.8 / 116.1 / 110.2 dB | 122.0 / 121.3 / 117.1 dB | 112.9 / 112.2 / 108.0 dB |
| DeEcho-DeReverb | CPU / Vulkan / CUDA | 119.5 / 121.7 / 77.6 dB | 120.6 / 119.9 / 76.4 dB | 106.1 / 105.4 / 61.8 dB |

DeEcho-DeReverb on CUDA is the one outlier, still far below audibility. DeNoise also passes on Apple
Metal (M4); the De-Echo models run the same graph there but have not been measured on Metal yet.

SNR here is against reference outputs, not against clean recordings: it shows the port computes what
UVR's network computes, not how well a model denoises.

## Sources

Checkpoints from the [`seanghay/uvr_models`](https://huggingface.co/seanghay/uvr_models) mirror at
revision `6f4fc0c`, SHA256-verified by the converter; UVR modelparams and network code from
`Anjok07/ultimatevocalremovergui` at `a5f88453bfb2b38b05a965bcf67727243e0cbf19`.

| file | sha256 |
|---|---|
| `UVR-DeNoise-Lite.pth` | `0023492fe98c406817b5253965de19ede65d1c147db015a3a428f07602e99571` |
| `UVR-DeNoise.pth` | `5addf43ece5bddd18da9f575a02d7ffdb32342414e6ad7ac8d1dd7a04138a628` |
| `UVR-De-Echo-Normal.pth` | `b849dd575643b075c257fb7a96c2ef5a79d7a5e7df74a2b319ad47118f1ee769` |
| `UVR-De-Echo-Aggressive.pth` | `1bd1d79d9c5d1b17d20f96f8a9f8aff1b55a83014f70712446bf420c0188e0a0` |
| `UVR-DeEcho-DeReverb.pth` | `e644028ec82865dc0fe082bc6fea85a43f7c71cfe375caee2da2d154aa661ee7` |

Converted with `tools/convert_vr.py --preset <denoise_lite|denoise|deecho_normal|deecho_aggressive|deecho_dereverb>`.
BatchNorm is folded into the convolution and dense weights, the training-only auxiliary head is
dropped, and the band layout and resampling filters are written as metadata. The architecture and
weights are otherwise unchanged. Conversion is deterministic: re-running it reproduces these files
byte for byte.

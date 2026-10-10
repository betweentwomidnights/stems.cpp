---
license: other
license_name: unknown
pipeline_tag: audio-to-audio
tags:
- audio-source-separation
- music-source-separation
- vocal-separation
- mdx-net
- uvr
- gguf
- stems.cpp
---

# Kim_Vocal_2 (MDX-Net) GGUF

> **No license is stated for these weights anywhere upstream.** Kim_Vocal_2 is distributed with
> [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)'s MDX-Net models and
> mirrored without one. It is republished here in GGUF form with that caveat. Do not assume any
> license, and if you hold the rights and want it removed, open a discussion on this repository.

Kim_Vocal_2, the MDX-Net vocal model from UVR's model list, converted to GGUF for
[stems.cpp](https://github.com/betweentwomidnights/stems.cpp), a C++/ggml stem separator that runs on
CPU, CUDA, Vulkan and Metal with no Python. It estimates **vocals**; the **instrumental** is the mix
minus the vocals, as in UVR.

| file | size |
|---|---:|
| `mdx_net_kim_vocal_2-17M-v1.0-F32.gguf` | 63.6 MiB |

## Download and run

```sh
git clone --recursive https://github.com/betweentwomidnights/stems.cpp && cd stems.cpp
./build.sh vulkan                          # or cpu | cuda | metal   (Windows: build.cmd)
./models.sh mdx_net_kim_vocal_2            # stems.cpp v0.1.4 and later
build-vulkan/bin/stems-split -m models/mdx_net_kim_vocal_2-17M-v1.0-F32.gguf -i song.wav -o stems/
```

`stems-server` finds it by model name (`"model": "mdx_net_kim_vocal_2"`).

## What the runtime does

The network is KUIELab's MDX-Net (ConvTDFNet), rebuilt from the ONNX graph: Conv/BatchNorm/ReLU
blocks, BatchNorm-after-Linear TDF layers and multiplicative skips, 5 scales of 3 blocks, 48
channels. The spectral path follows UVR's MDX separation at the GUI defaults: an n_fft 7680 /
hop 1024 STFT cropped to 3072 bins, Hann-windowed chunks of 261120 samples, the first three bins
zeroed, and UVR's 1.009 compensation. See stems.cpp's `docs/MDX.md`.

## Parity

SNR of the C++ output against the unchanged ONNX file run in onnxruntime with UVR's own STFT and
demix, on a 20 s clip (demucs' `test.mp3`), as seg (the first chunk) / full (the whole clip).
Measured 2026-10-09 with ggml `9d0d910b` and `4ad3b30b` (identical). CPU: Core Ultra 9 275HX.
Vulkan and CUDA: RTX 5070 Laptop. Cosine is 1.0000000 on every output.

| backend | seg vocals | full vocals | full instrumental | 20 s clip |
|---|---:|---:|---:|---:|
| CPU | 101.5 dB | 101.3 dB | 143.6 dB | 17.7 s |
| CUDA | 99.5 dB | 99.4 dB | 142.1 dB | 4.3 s |
| Vulkan | 97.0 dB | 97.1 dB | 140.1 dB | 4.6 s |

Not yet measured on Apple Metal; every op in the graph is one the other stems.cpp models already
run there.

## Sources

- Upstream: `Kim_Vocal_2.onnx` from the [`seanghay/uvr_models`](https://huggingface.co/seanghay/uvr_models)
  mirror at revision `6f4fc0cfb0717c9033ffed12471a53008f145b20`, sha256
  `ce74ef3b6a6024ce44211a07be9cf8bc6d87728cc852a68ab34eb8e58cde9c8b`. UVR's `model_data.json`
  supplies its parameters (compensate 1.009, dim_f 3072, dim_t 256, n_fft 7680, primary stem Vocals).
- Converted with stems.cpp's `tools/convert_mdx.py --preset kim_vocal_2`, which walks the ONNX graph
  node by node and checks every op, attribute and connection. Remaining BatchNorms are folded into the
  adjacent weights and tensors get structural names. The architecture and weights are otherwise
  unchanged. Conversion is deterministic: re-running it reproduces this file byte for byte.

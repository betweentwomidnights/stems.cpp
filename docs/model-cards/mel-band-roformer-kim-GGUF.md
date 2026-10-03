---
license: mit
pipeline_tag: audio-to-audio
base_model: KimberleyJSN/melbandroformer
base_model_relation: quantized
tags:
- audio-source-separation
- music-source-separation
- vocal-separation
- mel-band-roformer
- roformer
- gguf
- stems.cpp
---

# Mel-Band RoFormer (Kim) GGUF

Kimberley Jensen's [Mel-Band RoFormer vocal model](https://huggingface.co/KimberleyJSN/melbandroformer)
converted to GGUF for [stems.cpp](https://github.com/betweentwomidnights/stems.cpp), a C++/ggml stem
separator that runs on CPU, CUDA, Vulkan and Metal with no Python. It estimates **vocals**; the
**instrumental** is the mix minus the vocals, as in the model's own inference script.

| file | size |
|---|---:|
| `mel_band_roformer_kim-0.2B-v1.0-F32.gguf` | 870 MiB |
| `mel_band_roformer_kim-0.2B-v1.0-F16.gguf` | 435 MiB |

## Download and run

```sh
git clone --recursive https://github.com/betweentwomidnights/stems.cpp && cd stems.cpp
./build.sh vulkan                         # or cpu | cuda | metal   (Windows: build.cmd)
./models.sh mel_band_roformer_kim         # F32; --encoding f16 for half the size
build-vulkan/bin/stems-split -m models/mel_band_roformer_kim-0.2B-v1.0-F32.gguf -i song.wav -o stems/
```

`stems-server` finds it by model name (`"model": "mel_band_roformer_kim"`).

## Parity with PyTorch

Each figure is the vocal-stem SNR of the C++ output against Kim's own `MelBandRoformer` run in
float32 on CPU, for a 20 s clip (demucs' `test.mp3`), as seg (the first 8 s chunk) / full
(chunked `demix_track` over the whole clip). Every result passes stems.cpp's parity check.
Measured 2026-10-02/03 on stems.cpp with ggml `f30f0cdc`. CPU, CUDA and Vulkan: Core Ultra 9
275HX / RTX 5070 Laptop. Metal: Apple M4, against torch refs dumped on the M4, where the M4's CPU
gets the same 61.5 / 74.4.

| encoding | CPU | CUDA | Vulkan | Metal | 20 s clip (Vulkan, RTX 5070) |
|---|---:|---:|---:|---:|---:|
| F32 | 57.0 / 72.7 | 57.0 / 72.7 | 57.0 / 72.7 | 61.5 / 74.4 | 6.5 s |
| F16 | 45.4 / 58.2 | 46.9 / 61.0 | 47.6 / 59.8 | | 5.1 s |

**F32 is the reference.** F16 halves every 2D matmul weight, halving the file, and costs about
13 dB while staying more than 58 dB below the vocal on the full track. The seg figures are lower
than the full ones because the first chunk is mostly reflected intro where the vocal is near
silence (largest absolute error 4e-8).

## Sources and license

- Upstream: [KimberleyJSN/melbandroformer](https://huggingface.co/KimberleyJSN/melbandroformer),
  `MelBandRoformer.ckpt` at revision `ac9b0614ab3cd7f77219e18ba494dfd93956c348`.
- Converted with stems.cpp's `tools/convert_roformer.py --preset kim`. Tensors are renamed, the
  mel band layout is written as plain index lists (so the runtime needs no librosa), and the
  rotary frequencies are checked and stored as a theta. The architecture and weights are
  otherwise unchanged.
- The upstream repository declares the MIT License in its metadata but ships no license file and
  names no copyright holder. `LICENSE` therefore carries the standard MIT text attributed to the
  uploading account, and `NOTICE` says so. `SHA256SUMS` lists every GGUF.

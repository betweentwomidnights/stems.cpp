---
license: other
license_name: unknown
pipeline_tag: audio-to-audio
tags:
- audio-source-separation
- drum-separation
- mdx23c
- gguf
- stems.cpp
---

# DrumSep (MDX23C, aufr33 & jarredou) GGUF

> **No license is stated for these weights anywhere upstream.** aufr33 and jarredou's DrumSep model
> was released without one; its original release page is no longer online, and the MSST-WebUI
> mirror it is taken from states none for the weights (its own code is AGPL-3.0). It is republished
> here in GGUF form with that caveat. Do not assume any license, and if you hold the rights and want
> it removed, open a discussion on this repository.

aufr33 and jarredou's six-stem drum separation model (MDX23C, `ep_141`, SDR 10.8059), converted to
GGUF for [stems.cpp](https://github.com/betweentwomidnights/stems.cpp), a C++/ggml stem separator that
runs on CPU, CUDA, Vulkan and Metal with no Python. It splits a drum track into **kick**, **snare**,
**toms**, **hh**, **ride** and **crash** in one pass.

| file | size |
|---|---:|
| `mdx23c_drumsep-0.1B-v1.0-F32.gguf` | 417.3 MiB |

## Download and run

```sh
git clone --recursive https://github.com/betweentwomidnights/stems.cpp && cd stems.cpp
./build.sh vulkan                          # or cpu | cuda | metal   (Windows: build.cmd)
./models.sh mdx23c_drumsep                 # stems.cpp v0.1.4 and later
build-vulkan/bin/stems-split -m models/mdx23c_drumsep-0.1B-v1.0-F32.gguf -i drums.wav -o kit/
```

`stems-server` finds it by model name (`"model": "mdx23c_drumsep"`). It expects a drum stem, for
example the `drums` output of `htdemucs`.

## What the runtime does

The network is ZFTurbo's `TFC_TDF_net` (v3): InstanceNorm / exact GELU / convolution blocks with
1x1 shortcuts and bottleneck TDF layers, concatenated skips, 5 scales of 2 blocks, 128 channels and
4 frequency subbands, on an n_fft 2048 / hop 512 STFT cropped to 1024 bins. Chunking follows ZFTurbo's
generic `demix` with the config's 4-way overlap. See stems.cpp's `docs/MDX.md`.

## Parity

SNR of the C++ output against ZFTurbo's model code and `demix`, unchanged, run in float32 on CPU, on
a 20 s clip (demucs' `test.mp3`), as seg (the first chunk) / full (the whole clip), per stem.
Measured 2026-10-09 with ggml `9d0d910b` and `4ad3b30b` (identical). CPU: Core Ultra 9 275HX.
Vulkan and CUDA: RTX 5070 Laptop. Cosine is 1.0000000 on every stem.

| backend | seg, worst to best | full, worst to best | 20 s clip |
|---|---:|---:|---:|
| CPU | 94.7 - 123.9 dB | 115.3 - 130.8 dB | 72.9 s |
| CUDA | 94.4 - 123.0 dB | 116.3 - 130.5 dB | 8.3 s |
| Vulkan | 96.2 - 124.8 dB | 119.0 - 131.8 dB | 8.4 s |

Not yet measured on Apple Metal; every op in the graph is one the other stems.cpp models already
run there.

## Sources

- Upstream checkpoint: `aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt` from the
  [`Sucial/MSST-WebUI`](https://huggingface.co/Sucial/MSST-WebUI) mirror at revision
  `90b617b15bd0dc0b784f3d361faca1b51173fe44`, sha256
  `d2a4aa53eb584d21eead358a4e66d1882ad182911be018f052b5da73be9096d0`. Trained by aufr33 and jarredou.
- Configuration: the matching training YAML from MSST-WebUI's GitHub at
  `39287d46072d2d1d91d9f1518abb1dd9a930e3d7` (`configs_backup/multi_stem_models/`), sha256
  `1f019da093523d34b95912add62c5ab5467d5640c3361c0cbf7a3dcab2bdd4ab`.
- Converted with stems.cpp's `tools/convert_mdx.py --preset drumsep`. Tensor names stay PyTorch's and
  the configuration is stored as metadata. The architecture and weights are otherwise unchanged.
  Conversion is deterministic: re-running it reproduces this file byte for byte.

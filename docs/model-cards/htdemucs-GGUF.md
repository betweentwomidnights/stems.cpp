---
license: mit
pipeline_tag: audio-to-audio
tags:
- audio-source-separation
- music-source-separation
- htdemucs
- demucs
- gguf
- stems.cpp
---

# HTDemucs GGUF

Meta's Hybrid Transformer Demucs (v4) converted to GGUF for
[stems.cpp](https://github.com/betweentwomidnights/stems.cpp), a C++/ggml stem separator that runs
on CPU, CUDA, Vulkan and Metal with no Python. The three published HTDemucs models are here:

| model | stems | files |
|---|---|---|
| `htdemucs` | drums, bass, other, vocals | `htdemucs-42M-v1.0-{F32,F16}.gguf` |
| `htdemucs_6s` | drums, bass, other, vocals, guitar, piano | `htdemucs_6s-27M-v1.0-{F32,F16}.gguf` |
| `htdemucs_ft` | drums, bass, other, vocals (a bag of four fine-tuned models, one per stem) | `htdemucs_ft-4x42M-v1.0-{F32,F16}.gguf` |

| file | size |
|---|---:|
| `htdemucs-42M-v1.0-F32.gguf` | 160 MiB |
| `htdemucs-42M-v1.0-F16.gguf` | 100 MiB |
| `htdemucs_6s-27M-v1.0-F32.gguf` | 104 MiB |
| `htdemucs_6s-27M-v1.0-F16.gguf` | 70 MiB |
| `htdemucs_ft-4x42M-v1.0-F32.gguf` | 640 MiB |
| `htdemucs_ft-4x42M-v1.0-F16.gguf` | 400 MiB |

## Download and run

```sh
git clone --recursive https://github.com/betweentwomidnights/stems.cpp && cd stems.cpp
./build.sh vulkan                   # or cpu | cuda | metal   (Windows: build.cmd)
./models.sh htdemucs_6s             # F32; --encoding f16 for the smaller files
build-vulkan/bin/stems-split -m models/htdemucs_6s-27M-v1.0-F32.gguf -i song.wav -o stems/
```

`stems-server` finds these files by model name (`"model": "htdemucs_6s"`); see the stems.cpp README.

## Parity with PyTorch

Each figure is the SNR of the C++ output against `demucs 4.0.1` in float32, full separation of a
20 s clip (demucs' `test.mp3`), for the worst stem of each model. Every stem of every model passes
stems.cpp's parity check (cosine 0.9999 or better). Measured 2026-10-02/03 on stems.cpp with ggml
`f30f0cdc`. CPU, CUDA and Vulkan: Core Ultra 9 275HX / RTX 5070 Laptop. Metal: Apple M4, against
torch refs dumped on the M4.

| model | encoding | CPU | CUDA | Vulkan | Metal |
|---|---|---:|---:|---:|---:|
| htdemucs | F32 | 78.4 | 73.5 | 78.3 | 70.7 |
| htdemucs | F16 | 56.9 | 73.5 | 56.4 | |
| htdemucs_6s | F32 | 75.3 | 68.5 | 75.2 | 74.3 |
| htdemucs_6s | F16 | 53.9 | 68.5 | 55.5 | |
| htdemucs_ft | F32 | 70.7 | 69.5 | 70.6 | 67.1 |
| htdemucs_ft | F16 | 63.5 | 69.5 | 64.4 | |

**F32 is the reference.** F16 stores only the cross-transformer's matmul weights in half
precision, with the convolutions kept F32. It costs about 20 dB on CPU and Vulkan and nothing
measurable on CUDA (whose F32 matmuls already run as TF32), and stays more than 50 dB below every
stem. It is about 40% smaller, not faster. The per-stem tables are in stems.cpp's `docs/PARITY.md`.

## Sources and license

- Upstream: [facebookresearch/demucs](https://github.com/facebookresearch/demucs), `demucs 4.0.1`.
  Checkpoints: htdemucs `955717e8-8726e21a.th`; htdemucs_6s `5c90dfd2-34c22ccb.th`; htdemucs_ft
  `f7e0c4bc-ba3fe64a.th`, `d12395a8-e57c48e6.th`, `92cfc3b6-ef3bcb9c.th`, `04573f0d-f3cf25b2.th`.
- Converted with stems.cpp's `tools/convert_htdemucs.py`. Tensors are renamed, packed attention
  projections are split into q/k/v, constants are folded into metadata, and the F16 files halve
  the transformer's matmul weights. The architecture and weights are otherwise unchanged; every
  GGUF records its upstream checkpoint in `general.source.file`.
- Code and weights are MIT-licensed by Meta Platforms, Inc. and affiliates; see `LICENSE` and
  `NOTICE`. `SHA256SUMS` lists every GGUF.

If you use HTDemucs, cite *Hybrid Transformers for Music Source Separation* (Rouard, Massa,
Défossez, ICASSP 2023).

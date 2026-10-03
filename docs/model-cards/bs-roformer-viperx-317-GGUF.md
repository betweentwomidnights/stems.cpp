---
license: other
license_name: unknown
pipeline_tag: audio-to-audio
tags:
- audio-source-separation
- music-source-separation
- vocal-separation
- bs-roformer
- roformer
- gguf
- stems.cpp
---

# BS-RoFormer (viperx ep_317) GGUF

> **No license is stated for these weights anywhere upstream.** viperx's checkpoint is distributed
> through Ultimate Vocal Remover's model repository without one. It is republished here in GGUF form
> with that caveat. Do not assume any license, and if you hold the rights and want it removed,
> open a discussion on this repository.

viperx's Band-Split RoFormer vocal model (`model_bs_roformer_ep_317_sdr_12.9755`), converted to GGUF
for [stems.cpp](https://github.com/betweentwomidnights/stems.cpp), a C++/ggml stem separator that
runs on CPU, CUDA, Vulkan and Metal with no Python. It estimates **vocals**; the **instrumental**
is the mix minus the vocals.

| file | size |
|---|---:|
| `bs_roformer_viperx_317-0.2B-v1.0-F32.gguf` | 609 MiB |

## Download and run

```sh
git clone --recursive https://github.com/betweentwomidnights/stems.cpp && cd stems.cpp
./build.sh vulkan                          # or cpu | cuda | metal   (Windows: build.cmd)
./models.sh bs_roformer_viperx_317
build-vulkan/bin/stems-split -m models/bs_roformer_viperx_317-0.2B-v1.0-F32.gguf -i song.wav -o stems/
```

`stems-server` finds it by model name (`"model": "bs_roformer_viperx_317"`).

## Parity with PyTorch

Each figure is the vocal-stem SNR of the C++ output against ZFTurbo's MSST `BSRoformer` run in
float32 on CPU, for a 20 s clip (demucs' `test.mp3`), as seg (the first 8 s chunk) / full
(chunked `demix_track`). Every result passes stems.cpp's parity check. Measured 2026-10-02/03 on
stems.cpp with ggml `f30f0cdc`. CPU, CUDA and Vulkan: Core Ultra 9 275HX / RTX 5070 Laptop.
Metal: Apple M4, against torch refs dumped on the M4; that machine's CPU gets 61.8 / 50.6 against
the same refs, so Metal is 1.8 dB under it on the full track, which is float32 accumulation over
12 layers.

| encoding | CPU | CUDA | Vulkan | Metal | 20 s clip (Vulkan, RTX 5070) |
|---|---:|---:|---:|---:|---:|
| F32 | 69.6 / 65.2 | 69.5 / 62.4 | 69.6 / 63.1 | 61.8 / 48.8 | 14.6 s |

**F16 is not published for this model.** It was measured (44.2 / 34.9 dB on CPU, 45.1 / 36.3 on
CUDA, 44.0 / 31.9 on Vulkan). That clears stems.cpp's cosine check, but it loses about 30 dB and
lands where Vulkan was before ggml learned to honour `GGML_PREC_F32`, a level stems.cpp treated
as a bug. Mel-Band (Kim) and HTDemucs lose much less at F16 and are published at both encodings.

## Sources and license

- Upstream checkpoint: [TRvlvr/model_repo](https://github.com/TRvlvr/model_repo) release
  `all_public_uvr_models`, file `model_bs_roformer_ep_317_sdr_12.9755.ckpt`, sha256
  `5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa`. Trained by viperx.
- Configuration: ZFTurbo's [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
  (MIT), `configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml`.
- Converted with stems.cpp's `tools/convert_roformer.py --preset viperx`. Tensors are renamed, the
  band layout is written as plain index lists, and the rotary frequencies are checked and stored
  as a theta. The architecture and weights are otherwise unchanged.
- **No license.** There is no `LICENSE` file here because upstream provides none; `NOTICE`
  records the provenance and the caveat. `SHA256SUMS` lists the GGUF.

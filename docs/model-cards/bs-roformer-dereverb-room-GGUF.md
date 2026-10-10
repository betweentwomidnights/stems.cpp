---
license: gpl-3.0
pipeline_tag: audio-to-audio
base_model: anvuew/dereverb_room
tags:
- audio-source-separation
- dereverberation
- bs-roformer
- roformer
- gguf
- stems.cpp
---

# Room dereverb BS-RoFormer (anvuew) GGUF

[anvuew's `dereverb_room`](https://huggingface.co/anvuew/dereverb_room), a mono Band-Split RoFormer
that removes room reverb from vocals, converted to GGUF for
[stems.cpp](https://github.com/betweentwomidnights/stems.cpp), a C++/ggml stem separator that runs on
CPU, CUDA, Vulkan and Metal with no Python. It estimates the dry vocal (**noreverb**); **reverb** is
the mix minus that estimate. Like the upstream model, these weights are licensed under the
**GNU GPL v3** (see `LICENSE`).

| file | size |
|---|---:|
| `bs_roformer_dereverb_room-29M-v1.0-F32.gguf` | 112.3 MiB |

## Download and run

```sh
git clone --recursive https://github.com/betweentwomidnights/stems.cpp && cd stems.cpp
./build.sh vulkan                          # or cpu | cuda | metal   (Windows: build.cmd)
./models.sh bs_roformer_dereverb_room      # stems.cpp v0.1.4 and later
build-vulkan/bin/stems-split -m models/bs_roformer_dereverb_room-29M-v1.0-F32.gguf -i vocal.wav -o dry/ --stems noreverb
```

`stems-server` finds it by model name (`"model": "bs_roformer_dereverb_room"`).

The network is mono. stems.cpp runs it once per channel of a stereo input and returns stereo stems,
so a stereo separation costs twice a mono one, and each output channel is exactly what separating
that channel alone gives. A mono input comes back as two identical channels.

## Parity with PyTorch

SNR of the C++ output against ZFTurbo's MSST `BSRoformer` with the upstream config, run in float32
on CPU, on a mono downmix of a 20 s clip (demucs' `test.mp3`): seg is the first 8.7 s chunk, full is
the chunked `demix_track` over the whole clip. Measured 2026-10-09 with ggml `9d0d910b` on a Core Ultra
9 275HX and an RTX 5070 Laptop.

| backend | seg cos / SNR | full cos / SNR | 20 s mono clip |
|---|---:|---:|---:|
| CPU | 0.9999997 / 62.4 dB | 0.9999998 / 64.6 dB | 51.4 s |
| CUDA | 0.9999997 / 62.4 dB | 0.9999998 / 64.6 dB | 6.4 s |
| Vulkan | 0.9999997 / 62.4 dB | 0.9999998 / 64.6 dB | 6.3 s |

Not yet measured on Apple Metal. It runs the same BS-RoFormer graph as the viperx model, which
passes there.

## Sources, changes and license

- Upstream: [anvuew/dereverb_room](https://huggingface.co/anvuew/dereverb_room) at revision
  `0b85f5b80b7f779b2dfe80f33a1b35b38af9376d`, licensed GPL-3.0. Checkpoint
  `dereverb_room_anvuew_sdr_13.7432.ckpt`, sha256
  `2edec521f09e26341c1923dc82c8c52dbc86478b42b9999f679535743c970cb3`; config
  `dereverb_room_anvuew.yaml`, sha256 `c37e3039521d79cd1daff129857f69fa80c6a1f383a0fe8cda757f2dfc5032f8`.
- **Modified** on 2026-10-09 by converting it to GGUF with stems.cpp's
  `tools/convert_roformer.py --preset dereverb_room`: tensors are renamed, the band layout is written
  as index lists, the rotary frequencies are checked and stored as a theta, and the configuration is
  stored as metadata. The architecture and weights are otherwise unchanged, and the conversion is
  deterministic.
- This converted work is distributed under the same GNU General Public License v3; `LICENSE` holds
  the full text. The upstream checkpoint and the converter that produced these files are its
  corresponding source.

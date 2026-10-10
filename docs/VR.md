# UVR 5.1 VR family

`vr_cascaded` architecture dispatch uses the existing Separator/C ABI, CLI and service.
The denoise, de-echo and de-reverb models share the same implementation:

| model | outputs (primary, secondary) | network capacity (`nout` / `nout_lstm`) | magnitude bins | bands | GGUF |
|---|---|---|---|---|---|
| `uvr_denoise_lite` | noise, denoised | 16 / 128 | 1024 + extra bin | 1 | 16.8 MiB F32 |
| `uvr_denoise` | noise, denoised | 48 / 128 | 672 + extra bin | 4 | 120.8 MiB F32 |
| `uvr_deecho_normal` | no_echo, echo | 48 / 128 | 672 + extra bin | 4 | 120.8 MiB F32 |
| `uvr_deecho_aggressive` | echo, no_echo | 48 / 128 | 672 + extra bin | 4 | 120.8 MiB F32 |
| `uvr_deecho_dereverb` | no_reverb, reverb | 64 / 128 | 672 + extra bin | 4 | 212.8 MiB F32 |

The primary output is the one the network's mask selects, as in UVR's `model_data.json`; the
secondary is its complement. That is why the two De-Echo models list their outputs in opposite
orders. UVR's registry gives no capacity for DeEcho-DeReverb; its weights are the network's
default 64 / 128, and the converter reads capacity from the weights for every model.

The network has five encoder/decoder branches arranged as three cascaded stages. Each branch
has a dilated ASPP module and a bidirectional LSTM. Convolutions, interpolation, pooling,
LSTM gates and dense layers all run in ggml on the selected backend. BatchNorm is folded into
weights at conversion, and all matmuls request F32 accumulation. Recurrent states are zeroed
before every window; no state leaks between windows or requests. The runtime validates every
required tensor's type and dimensions before building a graph.

## Convert and run

```sh
python tools/convert_vr.py --preset denoise_lite models/
python tools/convert_vr.py --preset denoise models/
python tools/convert_vr.py --preset deecho_normal models/
python tools/convert_vr.py --preset deecho_aggressive models/
python tools/convert_vr.py --preset deecho_dereverb models/
stems-split -m models/uvr_denoise-32M-v1.0-F32.gguf -i input.wav -o output/ --float32
```

Python dependencies: torch, numpy, scipy, gguf. Checkpoints are downloaded to `models/src/`,
which is ignored by Git, and verified by SHA256. `--ckpt` reuses a local checkpoint. Preset
source files are pinned to the seanghay mirror revision `6f4fc0c`; UVR modelparams and reference
network code are pinned to `a5f88453bfb2b38b05a965bcf67727243e0cbf19`.

The service recognizes all five model IDs and resolves their locally converted GGUFs. Its request
format and the C ABI stay unchanged. Outputs are ordered as in the table above.
`--stems denoised` (or `no_echo`, `no_reverb`) selects the clean output but still runs the
complete network.

The F32 GGUFs are published at [thepatch/uvr-vr-GGUF](https://huggingface.co/thepatch/uvr-vr-GGUF)
(`models.sh`/`models.cmd`); there is no F16 or quantized version. No license is stated upstream:
the converter records `general.license=other` unless an exact license is supplied, and the
repository carries the same caveat as viperx's (docs/DISTRIBUTION.md).

## Portable spectral contract

The network is the upstream UVR 5.1 CascadedNet. The spectral path is explicitly defined so
desktop and mobile produce the same result:

* Stereo input at 44.1 kHz (the existing callers handle other sample rates and channel counts).
* Periodic Hann STFT with centered **zero** padding; not torch's default reflect padding.
* Full DeNoise uses the four crops/FFT sizes/hops in UVR's `4band_v3.json`, including
  320/640/960-point FFTs. Bluestein uses the existing radix-2 FFT for those sizes.
* Rational polyphase resampling in both directions, using scipy's default Kaiser FIR.
  The converter stores the FIRs in the GGUF; the runtime needs no scipy/libsamplerate.
* UVR 5.1 linear prefilter and synthesis low/high-pass masks.
* Normalize magnitude by the whole track's maximum, run 256-frame windows by default,
  discard 64 frames at each edge, and concatenate the remaining 128-frame masks.
* Primary waveform uses the predicted mask; secondary uses `1-mask`. Both reconstruct
  the filtered band representation. Unrepresented high frequencies are not bypassed.
* Preserve the requested output length by padding the last partial STFT hop with zeros,
  as UVR's length matching does. Silence returns zero outputs without allocating a graph.

This is **not a promise of byte-identical output from the UVR GUI**. In particular UVR's
Windows synthesis normally uses `sinc_fastest`, and optional aggressiveness/TTA/artifact
merging/high-end mirroring are absent here. The native network is checked against unchanged
UVR; complete waveform parity is checked against the explicit portable Python pipeline in
`tools/dump_refs_vr.py`. Explicit overlap and nonzero shifts are rejected rather than ignored.

## Extending the family

```sh
python tools/convert_vr.py models/custom.gguf --ckpt model.pth --params modelparams.json \
  --name vr_custom --primary no_echo --secondary echo
```

GGUF metadata stores source labels, network capacities, frequency crops, rates, FFT sizes,
hops, prefilter limits and inference window size. Compatible VR 5.1 checkpoints therefore
reuse the same implementation. The converter rejects unknown tensor families and unsupported
channel transformations. Older VR networks, alternate architectures and optional GUI
postprocessing require additional implementations; changing a filename does not enable them.

## Verification

The synthetic fixture is reproducible on a fresh checkout:

```sh
python tools/make_vr_fixture.py tests/data/vr-noisy.wav
python tools/dump_refs_vr.py denoise_lite tests/data/vr-noisy.wav tests/refs/vr_lite
python tools/dump_refs_vr.py denoise tests/data/vr-noisy.wav tests/refs/vr_full
```

Reference dependencies include librosa and soundfile in addition to the conversion dependencies.
The harness fetches its pinned UVR network code automatically. On M4, build with `./build.sh metal`,
run `ctest --test-dir build-metal --output-on-failure`, and compare both models using
`build-metal/bin/stems-vr-parity MODEL.gguf REFS tests/data/vr-noisy.wav cpu` and then `gpu`.
Record per-output SNR, timing and peak memory, and exercise the C ABI on the Metal build.

For the music excerpt, supply the existing Demucs stereo `test.wav` and create the excerpt with
`python tools/make_vr_fixture.py tests/data/vr-music.wav --source tests/data/test.wav`:

```sh
python tools/dump_refs_vr.py denoise_lite tests/data/vr-music.wav tests/refs/vr_lite_music
python tools/dump_refs_vr.py denoise tests/data/vr-music.wav tests/refs/vr_full_music
stems-vr-parity models/uvr_denoise_lite-4M-v1.0-F32.gguf tests/refs/vr_lite_music tests/data/vr-music.wav cpu
stems-vr-parity models/uvr_denoise-32M-v1.0-F32.gguf tests/refs/vr_full_music tests/data/vr-music.wav gpu
```

Measured 2026-10-07 on a 4.003 s excerpt of the existing Demucs `test.wav`, stereo 44.1 kHz.
CPU: Core Ultra 9 275HX, 4 threads. Vulkan: RTX 5070 Laptop. SNR compares against references,
not against a clean recording; it establishes implementation correctness, not denoising quality.

| model / backend | mask SNR | noise SNR | denoised SNR | separation time |
|---|---|---|---|---|
| Lite / CPU | 90.8 dB | 110.3 dB | 138.9 dB | 1.31 s |
| Full / CPU | 112.3 dB | 115.1 dB | 135.1 dB | 6.15 s |
| Lite / Vulkan | 104.9 dB | 112.0 dB | 139.2 dB | 0.45 s |
| Full / Vulkan | 114.1 dB | 115.8 dB | 135.1 dB | 1.20 s |
| Lite / CUDA | 106.3 dB | 114.2 dB | 139.4 dB | 0.52 s |
| Full / CUDA | 111.4 dB | 112.7 dB | 135.1 dB | 1.01 s |

Timings exclude model loading and initial Vulkan shader compilation, but include graph creation,
preprocessing, every window and reconstruction. Both CPU models also pass network and complete
waveform parity on that excerpt and on a 3.003 s synthetic noisy stereo fixture. Silence and
pre-inference cancellation are checked by the parity tool. The model-free VR DSP test covers
non-power-of-two FFTs, spectral round trips, filter endpoints and rational resampling alignment.

Both models also pass the C ABI smoke test, return both stems with progress callbacks, and
produce the same float WAVs as the CLI on Vulkan. Cancellation after a completed window works.

The three De-Echo models were checked the same way against unchanged UVR on the music excerpt
(2026-10-09, ggml `9d0d910b`, same machine), from preset conversions:

```sh
python tools/dump_refs_vr.py deecho_normal tests/data/vr-music.wav tests/refs/vr_deecho_normal_music
stems-vr-parity models/uvr_deecho_normal-32M-v1.0-F32.gguf tests/refs/vr_deecho_normal_music tests/data/vr-music.wav gpu
```

| model / backend | mask SNR | primary SNR | secondary SNR |
|---|---|---|---|
| De-Echo Normal / CPU | 118.7 dB | 123.1 dB (no_echo) | 110.7 dB (echo) |
| De-Echo Normal / Vulkan | 119.3 dB | 122.7 dB | 110.2 dB |
| De-Echo Aggressive / CPU | 114.8 dB | 122.0 dB (echo) | 112.9 dB (no_echo) |
| De-Echo Aggressive / Vulkan | 116.1 dB | 121.3 dB | 112.2 dB |
| DeEcho-DeReverb / CPU | 119.5 dB | 120.6 dB (no_reverb) | 106.1 dB (reverb) |
| DeEcho-DeReverb / Vulkan | 121.7 dB | 119.9 dB | 105.4 dB |
| De-Echo Normal / CUDA | 110.8 dB | 115.7 dB | 103.0 dB |
| De-Echo Aggressive / CUDA | 110.2 dB | 117.1 dB | 108.0 dB |
| DeEcho-DeReverb / CUDA | 77.6 dB | 76.4 dB | 61.8 dB |

Every run passes the tool's silence and cancellation checks. Cosines are 1.000000000 except
DeEcho-DeReverb on CUDA (0.99999999 / 0.99999999 / 0.9999997). That model is the only 64-channel
network, and CUDA alone loses about 40 dB on it, most likely because a matmul at its shapes runs
on a reduced-precision tensor-core path despite the F32 precision request. The 48-channel models
and DeNoise on the same build are unaffected. 62 dB is far below audibility, but it is the
largest gap in this table and is worth tracing to the kernel.

On an Apple M4 (32 GB, ggml `4ad3b30b`, 2026-10-10), with references dumped on that machine
from the same music excerpt, all three pass on CPU and Metal with cosine 1.000000000:

| model / backend | mask SNR | primary SNR | secondary SNR | separation time | peak memory |
|---|---|---|---|---|---|
| De-Echo Normal / CPU | 118.5 dB | 122.7 dB | 110.2 dB | 8.83 s | 1.34 GB |
| De-Echo Normal / Metal | 118.8 dB | 122.5 dB | 110.0 dB | 1.36 s | 381 MB |
| De-Echo Aggressive / CPU | 114.8 dB | 122.0 dB | 113.0 dB | 9.12 s | 1.33 GB |
| De-Echo Aggressive / Metal | 116.0 dB | 121.8 dB | 112.7 dB | 1.45 s | 375 MB |
| DeEcho-DeReverb / CPU | 121.0 dB | 120.4 dB | 105.9 dB | 13.03 s | 1.82 GB |
| DeEcho-DeReverb / Metal | 121.9 dB | 120.3 dB | 105.8 dB | 1.77 s | 484 MB |

Metal does not show CUDA's loss on the 64-channel DeEcho-DeReverb. On Metal the C ABI returns
both stems with progress for all three, cancels after the first window, and writes the same
float WAVs as the CLI.

Metal and mobile timing/memory measurements must be recorded separately; desktop GPU results
do not establish real-time audio-callback or phone suitability.
The current spectral buffers scale with recording length, so phone evaluation must include
memory measurements at the app's maximum recording duration, not just checkpoint size.

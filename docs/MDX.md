# TFC-TDF models: MDX-Net and MDX23C

`tfc_tdf` is one architecture for the U-Nets of TFC-TDF blocks that UVR (MDX-Net) and ZFTurbo's
Music-Source-Separation-Training (MDX23C) run on a complex spectrogram. `tfc_tdf.variant` picks
the block structure and `tfc_tdf.chunking` picks the reference pipeline:

| model | variant | outputs | n_fft / hop / dim_f / dim_t | network | GGUF |
|---|---|---|---|---|---|
| `mdx_net_kim_vocal_2` | `mdx_net` | vocals, instrumental | 7680 / 1024 / 3072 / 256 | 5 scales x 3 convs, 48 + 48 channels, BatchNorm + ReLU | 63.6 MiB F32 |
| `mdx23c_drumsep` | `mdx23c` | kick, snare, toms, hh, ride, crash | 2048 / 512 / 1024 / 256 | 5 scales x 2 blocks, 128 + 128 channels, 4 subbands, InstanceNorm + GELU | 417.3 MiB F32 |

The two variants differ in more than their sizes:

* **`mdx_net`** is KUIELab's ConvTDFNet as UVR distributes it, an ONNX file with no PyTorch
  source. Post-activation Conv + BatchNorm + ReLU (the exporter folded every BatchNorm that
  follows a convolution); a TDF is Linear over frequency, BatchNorm, ReLU, twice, added back;
  decoders upsample with a transposed convolution and *multiply* by the encoder output.
  One stem: the instrumental is the mix minus the vocals.
* **`mdx23c`** is ZFTurbo's `TFC_TDF_net` (v3). Pre-activation InstanceNorm / GELU / Conv, a 1x1
  shortcut around every block, concatenated skips, the spectrogram split into 4 frequency
  subbands stacked as channels, the first convolution's output multiplied back in before the
  last two 1x1 convolutions, and all six stems from one pass.

Everything between the STFT and the iSTFT runs in ggml on the selected backend: convolutions
are im2col plus F32 matmuls, the 3x3 ones in bands of output rows so no im2col exceeds 128 MB
(Kim's first scale would otherwise need 1.4 GB); a stride == kernel transposed convolution is
one matmul plus two permutes; InstanceNorm is `ggml_norm` over each channel's plane; GELU is
the exact erf form. The STFT and iSTFT run on the host in double precision; Kim's 7680-point
transform uses the Bluestein FFT from the VR work. The runtime validates every tensor's shape
against the metadata before building a graph.

## Convert and run

```sh
python tools/convert_mdx.py --preset kim_vocal_2 models/
python tools/convert_mdx.py --preset drumsep models/
stems-split -m models/mdx_net_kim_vocal_2-17M-v1.0-F32.gguf -i song.wav -o stems/
stems-split -m models/mdx23c_drumsep-0.1B-v1.0-F32.gguf -i drums.wav -o drums/
```

Python dependencies: numpy and gguf, plus onnx (Kim) or torch and pyyaml (DrumSep). Sources are
downloaded to `models/src/` and verified by SHA256; `--src` reuses a local copy.

* **Kim_Vocal_2.onnx** from the seanghay mirror (revision `6f4fc0c`, the one the VR presets
  use), SHA256 `ce74ef3b…9c8b`. The converter walks the ONNX graph node by node and checks every
  op, attribute and connection against the ConvTDFNet pattern, so a different export fails
  loudly instead of converting wrong. It folds the BatchNorm after each transposed convolution
  into its weights, keeps each TDF BatchNorm as a per-channel scale and shift, and names the
  weights structurally (`src/tfc_tdf.cpp` lists the names). Settings come from UVR's
  `models/MDX_Net_Models/model_data/model_data.json` at `a5f88453`, entry `970b3f94…` (UVR keys
  it by the MD5 of the file's last 10 MB): compensate 1.009, dim_f 3072, dim_t 2^8,
  n_fft 7680, primary stem Vocals.
* **DrumSep** (`aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt`) from MSST-WebUI's
  Hugging Face mirror (revision `90b617b1`), SHA256 `d2a4aa53…96d0`, with its training YAML from
  MSST-WebUI's GitHub (`39287d46`), SHA256 `1f019da0…d4ab`. Tensor names stay PyTorch's.

Neither model has a stated license. Kim_Vocal_2 ships in UVR's model list without one, and
DrumSep's original release (`github.com/jarredou/models`) no longer exists; the mirrors state
none for the weights (MSST-WebUI's own code is AGPL-3.0). The converter records
`general.license=other`, and these GGUFs are not published: convert them locally.

The service knows both model IDs. `--stems vocals` (or `kick`, ...) selects outputs but still
runs the whole network.

## Pipelines

**Kim_Vocal_2 follows UVR's `SeperateMDX`** (`separate.py`) at the GUI's defaults, the path
that runs the ONNX file in onnxruntime:

* Prepend `n_fft / 2` zeros, append zeros up to a whole number of `chunk - n_fft` steps plus
  `n_fft / 2`, where `chunk = hop * (dim_t - 1)` = 261120 samples (5.9 s).
* Every `chunk - n_fft` samples ("Default" overlap), take one chunk, zero-padded at the end:
  torch STFT (periodic Hann, centered, reflect padding), keep the lowest `dim_f` bins, zero
  bins 0-2, network, zero-pad back to `n_fft / 2 + 1` bins, torch iSTFT.
* Weight each chunk's output by `np.hanning` (the symmetric window) of its unpadded length,
  so the final short chunk gets its own short window, as in UVR; divide by the summed windows.
* Drop the leading `n_fft / 2` samples, trim to the input length, multiply by `compensate`.
  The instrumental is the mix minus that.

An explicit `overlap` follows UVR's numeric overlap setting: a step of `(1 - overlap) * chunk`,
with no window at overlap 0. UVR's optional denoise pass, pitch shift and spectral inversion
are not implemented; nonzero shifts are rejected.

**DrumSep follows ZFTurbo's generic `demix`** (`utils/model_utils.py` at `84b1eac0`) with the
YAML's `num_overlap: 4` and batch size 1: reflect-pad `chunk - step` samples at each end,
chunks of `hop * (dim_t - 1)` = 130560 samples every quarter chunk, linear fades of a tenth of a
chunk except at the very first chunk's start and the very last chunk's end, a short final chunk
reflect-padded when it is longer than half a chunk and zero-padded otherwise. This differs
slightly from Kim's `demix_track`, which the RoFormers follow (which chunks lose their fade-out,
and the half-chunk threshold), so it is implemented separately. An explicit `overlap` sets the
step to `(1 - overlap) * chunk`, as for the RoFormers.

The MSST demix runs DrumSep under CUDA fp16 autocast when it has a GPU; the reference here runs
it in float32, as on CPU.

## Verification

```sh
python tools/convert_mdx.py --preset kim_vocal_2 models/
python tools/dump_refs_mdx.py kim_vocal_2 tests/data/test.wav tests/refs/mdx_net_kim_vocal_2
build/bin/stems-parity --model models/mdx_net_kim_vocal_2-17M-v1.0-F32.gguf \
    --refs tests/refs/mdx_net_kim_vocal_2 --wav tests/data/test.wav --min-cos 0.99999
python tools/convert_mdx.py --preset drumsep models/
python tools/dump_refs_mdx.py drumsep tests/data/test.wav tests/refs/mdx23c_drumsep
build/bin/stems-parity --model models/mdx23c_drumsep-0.1B-v1.0-F32.gguf \
    --refs tests/refs/mdx23c_drumsep --wav tests/data/test.wav --min-cos 0.99999
```

`dump_refs_mdx.py` needs numpy, soundfile and torch, plus onnx and onnxruntime (Kim) or pyyaml
and ml_collections (DrumSep). For Kim it runs the unchanged ONNX file in onnxruntime with UVR's
own STFT class, fetched at the pinned UVR revision, and a transcription of `SeperateMDX.demix`
and `run_model` (separate.py cannot be imported without the GUI). For DrumSep it runs ZFTurbo's
`mdx23c_tfc_tdf_v3.py` and `demix` unchanged, fetched at the pinned commit.

**seg** is one chunk exactly as the chunking feeds it, through STFT, network and iSTFT, before
windowing and compensation; **full** is the whole 20 s clip (the Demucs `test.wav`). The clip
is a full mix, not a drum stem: that does not matter for implementation parity, but DrumSep's
separations of it are not meaningful as drum stems.

Measured 2026-10-09 on Windows at ggml `9d0d910b` (Core Ultra 9 275HX, 24 threads; RTX 5070
Laptop GPU); every SNR is identical at `4ad3b30b`, the revision pinned now. Cosine is 1.0000000 on every output of every row, so the table gives SNR (dB).
Times are `separate` on the 20 s clip, excluding model load and Vulkan shader compilation.

| model / backend | seg | full (per output) | 20 s clip |
|---|---|---|---|
| Kim_Vocal_2 / CPU | 101.5 | vocals 101.3, instrumental 143.6 | 17.7 s |
| Kim_Vocal_2 / Vulkan | 97.0 | vocals 97.1, instrumental 140.1 | 4.6 s |
| Kim_Vocal_2 / CUDA | 99.5 | vocals 99.4, instrumental 142.1 | 4.3 s |
| DrumSep / CPU | 94.7 – 123.9 | kick 130.8, snare 126.8, toms 124.4, hh 119.9, ride 123.7, crash 115.3 | 72.9 s |
| DrumSep / Vulkan | 96.2 – 124.8 | kick 131.8, snare 128.2, toms 124.9, hh 123.5, ride 124.9, crash 119.0 | 8.4 s |
| DrumSep / CUDA | 94.4 – 123.0 | kick 130.5, snare 126.5, toms 123.6, hh 121.2, ride 123.1, crash 116.3 | 8.3 s |

The lowest seg number is always `crash`, whose stem is near silence on this clip (largest
absolute error 5e-8). ggml's direct `CONV_2D` was tried first for the 3x3 convolutions. It matched on CPU, but Kim fell
to 29 dB on Vulkan: with cooperative matrices its shader staged tiles in fp16 whatever the kernel
type. betweentwomidnights/ggml#16 (in the pinned `4ad3b30b`) keeps an F32-kernel `CONV_2D` in
fp32, and direct convolution then passes on every backend with the same SNRs. Banded im2col stays
because it is faster overall. Back to back on the 20 s clip, GPU otherwise idle:

| backend | Kim, im2col | Kim, direct | DrumSep, im2col | DrumSep, direct |
|---|---|---|---|---|
| CPU | 20.3 – 21.4 s | 24.5 – 24.8 s | 81.2 s | 78.2 s |
| Vulkan | 4.8 – 5.1 s | 3.5 – 5.0 s | 8.9 – 11.1 s | 7.4 – 8.6 s |
| CUDA | 4.6 – 7.1 s | 14.6 – 15.4 s | 9.1 – 9.2 s | 58.7 – 70.9 s |

ggml's CUDA `CONV_2D` is a plain direct kernel, 3–7x slower here than im2col into cuBLAS. On
Vulkan direct convolution is up to a quarter faster and needs no im2col memory, so picking it per
backend is a possible follow-up. This RTX 5070 reported `KHR_coopmat` (no coopmat2) in every run,
so the coopmat2 path of ggml#16 was not exercised here.

Every row passes cosine >= 0.99999 and SNR >= 50 dB on every output. Not yet measured on Metal;
every op in the graph (im2col, `mul_mat`, `norm`, `gelu_erf`, `pad`, `concat`, permutes) is
one the other models already run there.

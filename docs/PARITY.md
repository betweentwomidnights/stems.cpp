# Parity with PyTorch

Measured 2026-09-24 on CPU (Xeon E3-1535M v6) against `demucs 4.0.1` / torch 2.14 CPU.
Test audio: demucs' own public `test.mp3` (20.0 s), decoded to 44.1 kHz stereo float.

Two checks per model:

- **seg** is one raw `model(x)` on the first 7.8 s segment, exactly as `apply_model` feeds it.
- **full** is the whole `demucs` separate path: normalise by the mono mix, then
  `apply_model(split=True, overlap=0.25, shifts=0)`, then denormalise. That exercises the three
  overlapping segments, the triangular cross-fade, the centre-padding of the last segment
  and the trim.

`shifts=0` because demucs' default of one shift is a *random* offset and can't be compared
sample for sample. `--shifts N` is implemented the same way, with a seeded RNG.

SNR (dB) of the C++ output against torch, per stem:

| model | check | drums | bass | other | vocals | guitar | piano |
|---|---|---|---|---|---|---|---|
| htdemucs | seg | 111.4 | 85.9 | 124.4 | 79.0 | | |
| htdemucs | full | 118.2 | 85.0 | 124.5 | 78.7 | | |
| htdemucs_6s | seg | 118.0 | 78.9 | 125.4 | 76.1 | 78.9 | 81.3 |
| htdemucs_6s | full | 124.4 | 83.6 | 126.1 | 75.8 | 77.8 | 81.5 |
| htdemucs_ft | seg | 111.8 | 121.9 | 122.7 | 112.0 | | |
| htdemucs_ft | full | 117.3 | 81.0 | 124.3 | 71.1 | | |

Cosine similarity is 1.0000000 to seven places everywhere. The largest absolute sample error
is 1.2e-5. The spread between stems is set by how quiet the stem is in this clip, not by
anything in the port.

## Reproduce

```bash
python -m venv .venv-ref && .venv-ref/bin/pip install torch demucs==4.0.1 gguf soundfile numpy
curl -L -o tests/data/test.mp3 https://raw.githubusercontent.com/adefossez/demucs/main/test.mp3
ffmpeg -i tests/data/test.mp3 -ar 44100 -ac 2 -c:a pcm_f32le tests/data/test.wav
for m in htdemucs htdemucs_6s htdemucs_ft; do
  .venv-ref/bin/python tools/convert_htdemucs.py $m models/$m-f32.gguf
  .venv-ref/bin/python tools/dump_refs.py $m tests/data/test.wav tests/refs/$m
  build/bin/stems-parity --model models/$m-f32.gguf --refs tests/refs/$m --wav tests/data/test.wav
done
```

`stems-parity --dump DIR` also writes every encoder, transformer and decoder activation as raw
float32. The frequency-branch tensors are memory-identical to the torch arrays in `tests/refs`.
The two transformer taps are token-major, so transpose them before comparing.

## One bug worth knowing about

The first full-path run failed completely (cosine 0.3) while the single segment matched
exactly. The cause: the transformer's positional embeddings were uploaded once, when the graph
was built. ggml's graph allocator reuses an input tensor's memory once its last consumer has run,
so from the second segment on the embeddings were overwritten scratch. Every input is now
re-uploaded before each compute. Any ggml port that reuses a graph across calls has the same trap.

## Speed

On this CPU, `htdemucs` separates at about 0.45x realtime.

## CUDA (Quadro P4000, sm_61)

Measured 2026-09-25, same refs and test clip, `stems-parity --device gpu`. Every stem of every
model passes at cosine 1.0000000:

| model | check | drums | bass | other | vocals | guitar | piano | 20 s clip |
|---|---|---|---|---|---|---|---|---|
| htdemucs | full | 112.7 | 82.7 | 116.9 | 78.4 | | | 5.9 s |
| htdemucs_6s | full | 117.8 | 82.5 | 112.2 | 75.3 | 76.5 | 80.2 | 7.5 s |
| htdemucs_ft | full | 117.0 | 80.9 | 103.4 | 71.0 | | | 23.6 s |

`htdemucs` runs at about 3.4x realtime here, about 7.5x the CPU build.

### A stale build dir looks like a model bug

Each backend builds into its own directory, so rebuilding one leaves the others on old code.
On 2026-09-25 `build-cuda/` was still from before the positional-embedding fix above, and it
produced stems that were mostly noise above 8 kHz and did not sum back to the mix (0.4 dB).
The single-segment check still passed, because the bug only shows from the second segment on.
Only the `full` check catches it. `build.sh` now warns when another build dir is older than
the sources. Run `stems-parity --wav` on whichever build you are about to use.

## RoFormers

Measured 2026-10-02 on Windows (Core Ultra 9 275HX, RTX 5070 Laptop GPU), same 20 s test clip.
The reference is the model code each checkpoint was published with, run in float32 on CPU:
Kim's own `MelBandRoformer` for `mel_band_roformer_kim`, ZFTurbo's MSST `BSRoformer` for viperx's
`ep_317`. Chunking is `demix_track` (8 s chunks, `num_overlap: 2`, linear fades, reflect-padded
borders) in both.

- **seg** is one raw `model(x)` on the first chunk, exactly as `demix_track` feeds it.
- **full** is `demix_track` over the whole clip.

Both are single-stem vocal models, so there is one stem to compare. The `instrumental` the
runtime adds is mix minus vocals, computed in float.

| model | backend | seg cos / SNR | full cos / SNR | 20 s clip |
|---|---|---|---|---|
| mel_band_roformer_kim | CPU | 0.9999992 / 57.0 | 1.0000000 / 72.7 | 151 s |
| mel_band_roformer_kim | CUDA | 0.9999992 / 57.0 | 1.0000000 / 72.7 | 9.8 s |
| mel_band_roformer_kim | Vulkan | 0.9999992 / 57.0 | 1.0000000 / 72.7 | 6.6 s |
| mel_band_roformer_kim, F16 weights | CUDA | 0.9999914 / 46.9 | 0.9999998 / 61.0 | 10.0 s |
| mel_band_roformer_kim, F16 weights | Vulkan | 0.9999921 / 47.6 | 0.9999997 / 59.8 | 5.2 s |
| bs_roformer_viperx_317 | CPU | 0.9999999 / 69.6 | 0.9999999 / 65.2 | 411 s |
| bs_roformer_viperx_317 | CUDA | 0.9999999 / 69.5 | 0.9999999 / 62.4 | 22.4 s |
| bs_roformer_viperx_317 | Vulkan | 0.9999999 / 69.6 | 0.9999999 / 63.1 | 14.2 s |

Vulkan is RTX 5070 Laptop (coopmat2), ggml `07f9348a`. Before betweentwomidnights/ggml#7, Vulkan
rounded the operands of every F32 x F32 matmul to fp16 whatever `GGML_PREC_F32` said, and these
rows read 45.2 / 57.8 (Kim) and 44.2 / 33.6 dB (viperx). Getting fp32 back costs about 40% on
this card: Kim went from 4.7 to 6.6 s and viperx from 9.2 to 14.2 s. A caller that would
rather have the speed can drop `GGML_PREC_F32` on the matmuls.

The seg SNRs look low next to the full ones because the first chunk is mostly the reflected
intro, where the vocal stem is near silence: the largest absolute error there is 4e-8. F16
weights halve Kim's GGUF and stay far below anything audible. MSST and UVR run these models
under fp16 autocast on CUDA anyway.

### Two things the reference code does that the config doesn't say

- **Mask MLP depth.** `mask_estimator_depth: 2` builds three Linear layers in Kim's copy of the
  code and two in MSST's (`MLP` changed its meaning of `depth`). The converter counts the
  layers in the checkpoint and writes `roformer.mask_layers`, so the config's number is never trusted.
- **`zero_dc`.** MSST's BS- and Mel-Band RoFormer zero the DC bin of the masked spectrogram
  before the iSTFT (`zero_dc=True` by default); Kim's code does not. Without it viperx's first
  chunk matched at only cos 0.984. The converter writes `roformer.zero_dc`: on for MSST
  checkpoints, off for Kim's.

### Reproduce

```bash
.venv-ref/bin/pip install torch numpy soundfile librosa pyyaml ml_collections gguf huggingface_hub \
    einops==0.6.1 beartype==0.14.1 rotary_embedding_torch==0.3.5
python tools/convert_roformer.py --preset kim models/mel_band_roformer_kim-f32.gguf
python tools/dump_refs_roformer.py kim tests/data/test.wav tests/refs/mel_band_roformer_kim
build/bin/stems-parity --model models/mel_band_roformer_kim-f32.gguf \
    --refs tests/refs/mel_band_roformer_kim --wav tests/data/test.wav
```

For viperx, download `model_bs_roformer_ep_317_sdr_12.9755.ckpt` from UVR's model repo and its
config from MSST's `configs/viperx/`, then `convert_roformer.py --arch bs_roformer --ckpt ...
--config ... --name bs_roformer_viperx_317 --complement instrumental` and
`dump_refs_roformer.py viperx_bs_317 ... --ckpt ...`.

## Vulkan is not exact

ggml's Vulkan matmul shaders lose precision that CPU and CUDA keep, with or without cooperative
matrices (`GGML_VK_DISABLE_COOPMAT=1` made it worse). The RoFormers stay at 33–58 dB, but
HTDemucs does not hold up. On the same clip and GPU, `htdemucs` bass and vocals fall to
33–36 dB, and `htdemucs_6s` to 10–18 dB on vocals, guitar, piano and bass (cos 0.958 for
vocals). Under investigation; until then, use CUDA or CPU when the stems must be exact.

## Metal (Apple M4)

Measured 2026-10-02 on an Apple M4 Mac (32 GB), ggml `07f9348a`, same
refs and test clip, `stems-parity --device gpu`. CPU on the same machine matches the tables
above (Kim full 74.4 dB, `htdemucs` 78.5–122.7 dB), so every gap below is Metal's.

| model | check | drums | bass | other | vocals | guitar | piano | 20 s clip |
|---|---|---|---|---|---|---|---|---|
| htdemucs | full | 70.1 | 42.9 | 75.2 | 51.0 | | | 12.2 s |
| htdemucs_6s | seg | 65.8 | **35.8 (FAIL)** | 73.6 | 53.5 | 48.9 | 49.2 | |
| htdemucs_6s | full | 71.1 | 40.4 | 74.2 | 53.1 | 49.3 | 48.8 | 14.9 s |
| htdemucs_ft | full | 72.8 | 37.5 | 77.4 | 51.5 | | | 48.9 s |

| model | seg cos / SNR | full cos / SNR | 20 s clip |
|---|---|---|---|
| mel_band_roformer_kim | 0.9999923 / 47.9 | 0.9999994 / 59.1 | 27.5 s |
| bs_roformer_viperx_317 | 0.9999808 / 44.1 | 0.9999575 / 29.0 | 66.6 s |

Metal's `kernel_mul_mm_f32_f32` stages both operands as `half` and ignores `GGML_PREC_F32`:
the same fp16 rounding betweentwomidnights/ggml#7 removed from Vulkan, and the viperx numbers
look like pre-#7 Vulkan. It needs the same fix in ggml's Metal backend. Until then, use CPU on
a Mac when the stems must be exact. The M4 GPU is about 1.7x faster than its CPU on HTDemucs
and about 8.5x on Kim (CPU: 21 s and 235 s).

Before 2026-10-02 HTDemucs did not run on Metal at all, at any ggml pin:

- **Left padding.** Metal's `PAD` only pads on the right, and the transposed conv's
  overlap-add padded its second half on the left: `unsupported op 'PAD'`. It now pads on the
  right and rolls.
- **`group_norm` over ne3.** Metal's `group_norm` ignores ne3, so the frequency branch's
  per-row norms (one group over `[W, C, 1, H]`) were wrong from `enc0` on and reached NaN by
  `enc3`. They now run as H groups over `[W, C, H]`, the same norm.

Both are pure re-expressions: CPU stems are byte-identical to before (htdemucs, htdemucs_6s).

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

## Vulkan (RTX 5070 Laptop, coopmat2)

Measured 2026-10-02 with ggml `07f9348a`, Vulkan SDK 1.4.350.0, same refs and test clip. Every
stem of every model passes at cosine 1.0000000, and the numbers match the CPU table above:

| model | check | drums | bass | other | vocals | guitar | piano | 20 s clip |
|---|---|---|---|---|---|---|---|---|
| htdemucs | full | 118.1 | 84.9 | 124.2 | 78.3 | | | 2.2 s |
| htdemucs_6s | full | 124.3 | 82.8 | 125.8 | 75.2 | 76.9 | 80.7 | 2.3 s |
| htdemucs_ft | full | 117.5 | 80.7 | 124.2 | 70.6 | | | 8.5 s |

The same holds with `GGML_VK_DISABLE_COOPMAT2=1` (coopmat1), with both coopmat paths disabled
(scalar shaders), and on the laptop's Intel iGPU.

### fp16 operands in ggml-vulkan's F32 matmuls

Before ggml `217f0f2d` (betweentwomidnights/ggml#7), Vulkan failed parity on this card while CPU and CUDA passed:

| model | check | drums | bass | other | vocals | guitar | piano |
|---|---|---|---|---|---|---|---|
| htdemucs | full | 57.8 | 32.8 | 64.4 | 35.6 | | |
| htdemucs_6s | full | 68.2 | 17.6 | 68.1 | 10.5 | 11.5 | 15.5 |
| htdemucs_ft | full | 57.0 | 29.5 | 63.8 | 29.8 | | |

`stems-parity --dump` showed the error was already about 68 dB at `enc0`, the first
convolution, and stayed there through every layer. That points to per-op rounding, not
something accumulating. On any device with fp16 support, ggml-vulkan rounds both operands of
an F32 x F32 matmul to fp16. The scalar shader keeps its shared-memory tiles in fp16, the
coopmat shaders use fp16 matrices, and the coopmat2 path converts F32 inputs to fp16 up front.
`ggml_mul_mat_set_prec(GGML_PREC_F32)` only picked the accumulator. A -65 dB noise floor
relative to the mix is invisible on loud stems and ruins quiet ones. That is why vocals,
guitar and piano in this clip suffered most. It is also why disabling coopmat did not help:
the scalar fallback rounds to fp16 too. Only `GGML_VK_DISABLE_F16=1` together with both
coopmat switches restored parity.

The fork now honours `GGML_PREC_F32` on contiguous F32 x F32 matmuls. It runs them on the
fp32 build of the scalar shader, and every matmul in `htdemucs.cpp` already asks for that
precision. On this card it costs no measurable time for HTDemucs.

The pin, `f30f0cdc` (betweentwomidnights/ggml#11), is the commit shared by every consumer of the
fork. It is `07f9348a` (#10: #7 plus two Vulkan fixes HTDemucs does not depend on, #8 for the batch
stride of a matmul over a strided view and #9 for `PAD_REFLECT_1D`) with the Metal counterpart of #7
on top. #7 must not ship without #8. Before #7, coopmat2 copied every F32 input to a staging
buffer, and that copy hid #8's bug; `GGML_PREC_F32` now skips the copy. The table above was
re-measured at `07f9348a` and is identical to `217f0f2d` alone. #11 changes only Metal sources, so
it is unchanged at `f30f0cdc`.

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
| bs_roformer_dereverb_room (mono) | CPU | 0.9999997 / 62.4 | 0.9999998 / 64.6 | 51.4 s |
| bs_roformer_dereverb_room (mono) | Vulkan | 0.9999997 / 62.4 | 0.9999998 / 64.6 | 6.3 s |
| bs_roformer_dereverb_room (mono) | CUDA | 0.9999997 / 62.4 | 0.9999998 / 64.6 | 6.4 s |

anvuew's `dereverb_room` was measured on 2026-10-09 at ggml `9d0d910b`, against MSST's
`BSRoformer` on a mono downmix of the same clip (one channel, so one pass). Through
`load_separator` a mono model runs once per channel of a stereo input (dual mono), so a stereo
separation costs twice that; each output channel is bit-identical to running that channel alone.
It has not been measured on Metal yet; it runs the same BS-RoFormer graph as viperx (plain
`mul_mat` + `soft_max_ext` attention, no flash attention) at dim 128 and head size 16.

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

The room dereverb is mono, so its reference WAV is a mono downmix of `test.wav`; both tools fetch
the checkpoint and config at a pinned revision of `anvuew/dereverb_room`:

```bash
python -c "import soundfile as sf; d, sr = sf.read('tests/data/test.wav'); sf.write('tests/data/test-mono.wav', d.mean(1), sr, subtype='FLOAT')"
python tools/convert_roformer.py --preset dereverb_room models/
python tools/dump_refs_roformer.py dereverb_room tests/data/test-mono.wav tests/refs/bs_roformer_dereverb_room
build/bin/stems-parity --model models/bs_roformer_dereverb_room-29M-v1.0-F32.gguf \
    --refs tests/refs/bs_roformer_dereverb_room --wav tests/data/test-mono.wav
```

## TFC-TDF (MDX-Net, MDX23C)

Kim_Vocal_2 is checked against its unchanged ONNX file in onnxruntime inside UVR's MDX chunking;
DrumSep against ZFTurbo's own `TFC_TDF_net` and `demix`. Both pass on CPU, Vulkan and CUDA
(2026-10-09, ggml `9d0d910b`, identical at `4ad3b30b`) with cosine 1.0000000 on every output: Kim at 97–101 dB (vocals)
and 140–144 dB (instrumental), DrumSep at 115–132 dB per drum stem over the whole clip.
[docs/MDX.md](MDX.md) has the full table, the pipelines and how to reproduce them.

## Vulkan

The Vulkan rows above are at ggml `07f9348a` and later. Before betweentwomidnights/ggml#7, ggml's
Vulkan F32 matmuls rounded their operands to fp16 whatever precision was asked for. The RoFormers
then sat at 33–58 dB, and HTDemucs failed (see "fp16 operands in ggml-vulkan's F32 matmuls" above).
Both now match CPU. The cost on this card is about 40% on the RoFormers and nothing measurable on
HTDemucs.

## Metal (Apple M4)

Measured 2026-10-02/03 on an Apple M4 Mac (32 GB), same refs and test clip, `stems-parity --device
gpu`. The torch refs were dumped on the M4, and that machine's CPU build is the bar. It matches the
tables above (Kim full 74.4 dB, `htdemucs` 78.5–122.7 dB). The exception is viperx: CPU gets 50.6 dB
against ARM torch, versus 65.2 dB against the Windows/x86 refs.

**At ggml `f30f0cdc`** (betweentwomidnights/ggml#11, every model passes):

| model | worst stem, full | seg / full | 20 s clip |
|---|---|---|---|
| htdemucs | bass 70.7 | | 12.7 s |
| htdemucs_6s | bass 74.3 | | 15.5 s |
| htdemucs_ft | bass 67.1 | | |
| mel_band_roformer_kim | | 61.5 / 74.4 (= M4 CPU) | about 28 s |
| bs_roformer_viperx_317 | | 61.8 / 48.8 (M4 CPU 61.8 / 50.6) | about 76 s |

#11 is the Metal counterpart of ggml#7. Before it, `kernel_mul_mm_f32_f32` staged both operands as
`half` and ignored `GGML_PREC_F32`. #11 also cherry-picks an upstream fix for NORM/RMS_NORM rows
that leave a partial simdgroup: the band-split RMS norms here are 132, 264, 396... wide, and were
normalised with the wrong scale. Kim on Metal now matches the M4 CPU to the dB. viperx full sits
1.8 dB under it, which is float32 accumulation over 12 layers; every op compares at float32 noise.
It costs about 6–10% on the RoFormers and nothing on HTDemucs. The M4 GPU is about 1.7x its CPU on
HTDemucs and about 8.5x on Kim.

**At ggml `07f9348a`, before #11:**

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

Before 2026-10-02 HTDemucs did not run on Metal at all, at any ggml pin:

- **Left padding.** Metal's `PAD` only pads on the right, and the transposed conv's
  overlap-add padded its second half on the left: `unsupported op 'PAD'`. It now pads on the
  right and rolls.
- **`group_norm` over ne3.** Metal's `group_norm` ignores ne3, so the frequency branch's
  per-row norms (one group over `[W, C, 1, H]`) were wrong from `enc0` on and reached NaN by
  `enc3`. They now run as H groups over `[W, C, H]`, the same norm.

Both are pure re-expressions: CPU stems are byte-identical to before (htdemucs, htdemucs_6s).
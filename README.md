# stems.cpp

Stem separation for the gary ecosystem in C++ on [ggml](https://github.com/betweentwomidnights/ggml):
Meta's **HTDemucs v4** for 4 and 6 stems, and the **RoFormer** family (Mel-Band and Band-Split)
that UVR uses for vocals and instrumentals. No PyTorch, no Python at inference time.
A sibling of [sa3.cpp](https://github.com/betweentwomidnights/sa3.cpp) and
[audiocraft.cpp](https://github.com/betweentwomidnights/audiocraft.cpp): same ggml fork, same pin,
same build scripts, same service shape. It exists because gary4juce's Carey extract tab says
"if you have a stem separator it may work better". This is that separator.

Every model matches its PyTorch reference **sample for sample**. For HTDemucs that means
`demucs --shifts 0` on a 20 s song, through the whole split/overlap/trim path, at 71–126 dB SNR
per stem (float32 rounding). For the RoFormers it is the chunked `demix_track` the checkpoints
were published with, at 62–73 dB on CPU and CUDA:

| model | stems | params | F32 / F16 | Hugging Face |
|---|---|---|---|---|
| `htdemucs` | drums, bass, other, vocals | 42 M | 160 / 100 MiB | [thepatch/htdemucs-GGUF](https://huggingface.co/thepatch/htdemucs-GGUF) |
| `htdemucs_6s` | + guitar, piano | 27 M | 104 / 70 MiB | [thepatch/htdemucs-GGUF](https://huggingface.co/thepatch/htdemucs-GGUF) |
| `htdemucs_ft` | drums, bass, other, vocals (bag of 4 per-source fine-tunes, best quality, 4x the time) | 4x42 M | 640 / 400 MiB | [thepatch/htdemucs-GGUF](https://huggingface.co/thepatch/htdemucs-GGUF) |
| `mel_band_roformer_kim` | vocals, instrumental ([Kim's Mel-Band RoFormer](https://huggingface.co/KimberleyJSN/melbandroformer), MIT) | 228 M | 870 / 435 MiB | [thepatch/mel-band-roformer-kim-GGUF](https://huggingface.co/thepatch/mel-band-roformer-kim-GGUF) |
| `bs_roformer_viperx_317` | vocals, instrumental (viperx's BS-RoFormer ep_317; **no upstream license**, see its card) | 160 M | 609 MiB, F32 only | [thepatch/bs-roformer-viperx-317-GGUF](https://huggingface.co/thepatch/bs-roformer-viperx-317-GGUF) |

Files are named `<model>-<size>-v1.0-<F32|F16>.gguf`, e.g. `htdemucs-42M-v1.0-F32.gguf`.
[docs/DISTRIBUTION.md](docs/DISTRIBUTION.md) covers the naming, the repositories and the licenses.
F32 is the reference. F16 is published where it was measured to hold up (each model card has the
numbers).

See [docs/PARITY.md](docs/PARITY.md) for the numbers and how to reproduce them.

## Build

```bash
git clone --recurse-submodules https://github.com/betweentwomidnights/stems.cpp.git
cd stems.cpp
./build.sh cpu          # or: cuda | vulkan | metal | all   (windows: build.cmd cuda)
./models.sh             # htdemucs (F32); ./models.sh all, or name models; --encoding f16
```

`models.sh` (Windows: `models.cmd`) downloads from Hugging Face with curl. To convert the
checkpoints yourself instead (`pip install torch numpy pyyaml librosa gguf huggingface_hub demucs`):

```bash
python tools/convert_htdemucs.py htdemucs models/          # -> models/htdemucs-42M-v1.0-F32.gguf
python tools/convert_roformer.py --preset kim models/      # downloads Kim's checkpoint; --f16 for half the size
```

`build.sh` runs 4 compile jobs by default (`JOBS=8 ./build.sh cuda` to change it). An unbounded
`-j` on ggml's CUDA kernels used up 32 GB and took the build machine down.
Older nvcc with a newer gcc: `./build.sh cuda -DCMAKE_CUDA_HOST_COMPILER=g++-12`.

`ctest --test-dir build -C Release` runs the model-free tests: DSP round trips, the C ABI as a
host loads it, the `--version`/`--props` contract, and model-file resolution. The PyTorch parity
checks need references and are in [docs/PARITY.md](docs/PARITY.md).

Prebuilt Windows runtimes (core, CUDA, Vulkan, standalone) are attached to each
[GitHub release](https://github.com/betweentwomidnights/stems.cpp/releases);
[docs/RUNTIME_RELEASE.md](docs/RUNTIME_RELEASE.md) covers the packages and how they are built.

## Run

```bash
stems-split -m models/htdemucs-42M-v1.0-F32.gguf -i song.wav -o stems/                     # 4 stems
stems-split -m models/htdemucs-42M-v1.0-F32.gguf -i loop.wav -o stems/ --two-stems drums   # drums + no_drums
stems-split -m models/htdemucs_6s-27M-v1.0-F32.gguf -i song.wav -o stems/ --stems guitar,piano
stems-split -m models/mel_band_roformer_kim-0.2B-v1.0-F32.gguf -i song.wav -o stems/       # vocals + instrumental
```

Any WAV works: 16/24/32-bit or float, any sample rate (band-limited resample to 44.1 kHz),
mono or stereo. Stems come out 16-bit, scaled down only if they would clip (demucs'
`--clip-mode rescale`), or `--float32`. `--shifts N` averages N random time shifts, like demucs,
at N times the cost. `--overlap` defaults to the model's own: 0.25 for HTDemucs, 0.5 for the
RoFormers (their `num_overlap: 2`).

`no_X` from `--two-stems X` is the **sum of the other stems**. That is demucs' definition, and it
is not the same as mix minus X. A single-stem RoFormer is the other way round: it estimates
vocals, and `instrumental` **is** the mix minus vocals, as in the inference script Kim's model
was published with.

## Service

```bash
stems-server --port 8010 --models-dir models
```

```
GET  /health
GET  /props                      version, ggml devices, the file each model name resolves to
GET  /api/models
POST /separate                   JSON in, {success, model, sample_rate, stems: {name: b64 wav}}
POST /api/juce/separate_audio    -> {success, session_id}
GET  /api/juce/poll_status/<id>  -> {status, progress, separation_in_progress, stems on completion}
```

Request: `audio_data` (base64 WAV) and optionally `model` (`htdemucs` | `htdemucs_6s` |
`htdemucs_ft` | `mel_band_roformer_kim` | `bs_roformer_viperx_317`), `two_stems`, `stems` (list), `shifts`, `overlap`, `seed`, `float32`. The session
and poll shape is audiocraft.cpp's, so gary4juce can reuse the client code it already has for
terry. One job at a time. The model stays resident between requests and swaps when a different
one is asked for. `stems-server --version` and `--props` answer without binding a port or loading
a model; `STEMS_PORT`, `STEMS_HOST`, `STEMS_MODELS_DIR` and `STEMS_DEVICE` set the defaults.

## C ABI (libstems)

For embedding in a host (a JUCE/iPlug2 plugin, the iOS app, Tauri over FFI), the build also
produces `libstems.so` / `stems.dll` / `libstems.dylib`, or `libstems.a` with `-DSTEMS_STATIC=ON`
(forced on for iOS). The contract is `src/libstems_v1.h`. It is built the same way as sa3.cpp's
`libsa3_v1.h`: one exported symbol, `stems_get_api(STEMS_ABI_VERSION_1)`, returns a function
table. Structs are size-tagged (zero, set `size`, call the `*_init`, then fill in). The library
owns results and you free them with `result_free`. Progress and cancel are callbacks.

```c
const stems_api_v1* api = stems_get_api(STEMS_ABI_VERSION_1);
stems_context_config_v1 cfg = {sizeof cfg}; api->context_config_init(&cfg);
cfg.model_path = "models/htdemucs-42M-v1.0-F32.gguf";
stems_context* ctx; stems_error_v1 err = {sizeof err};
api->context_create(&cfg, &ctx, &err);

stems_request_v1 req = {sizeof req}; api->request_init(&req);
req.input.samples = buf; req.input.n_samples = n; req.input.n_channels = 2;
req.input.sample_rate = 48000; req.input.layout = STEMS_AUDIO_INTERLEAVED_V1;
stems_result_v1 res = {sizeof res}; api->result_init(&res);
api->separate(ctx, &req, &res, &err);   /* res.samples: [source][channel][sample] at 48 kHz */
api->result_free(&res);
api->context_destroy(ctx);
```

Input can be any rate, any channel count, planar or interleaved. Stems come back at the input's
sample rate and exact length, so a host never resamples. In a default build ggml is linked in
statically and kept private: `stems_get_api` is the only exported symbol, so a plugin can load
libstems next to libsa3 without their ggml copies colliding. The release packages instead ship
ggml and its backends as DLLs beside `stems.dll`, which loads its backends from its own folder;
see [docs/RUNTIME_RELEASE.md](docs/RUNTIME_RELEASE.md) for loading it from a plugin.
`tools/stems-libtest.c` is a complete example in plain C. Its output is byte-identical to
`stems-split --float32`.

## How it's put together

- `src/separator.h` is the interface every model architecture implements. `load_separator()` reads
  `general.architecture` from the GGUF and builds the matching one, so libstems, `stems-split` and
  `stems-server` never name a model class.
- `src/roformer.cpp` covers Mel-Band and BS-RoFormer in one class: band split, `depth` x (time
  transformer, freq transformer) with rotary attention, and a mask MLP per band, as one ggml graph
  per 8 s chunk. The two architectures differ only in which STFT bins form each band, and
  `tools/convert_roformer.py` writes that as plain index lists, so the runtime needs no librosa.
  Attention runs in groups of sequences so the score tensor stays under 256 MB.
- `src/htdemucs.cpp` holds the network as one ggml graph per 7.8 s segment: freq branch, time branch,
  and the 5-layer cross-transformer between them. Every conv is an F32 im2col plus matmul. Transposed convs
  are a matmul plus an explicit overlap-add, so only ops every backend has are used.
- `src/stft.h`: torch's `stft`/`istft` as HTDemucs calls them, on the host in double precision.
- `HTDemucs::separate` reproduces `demucs.apply.apply_model` (bag → shifts → split → segment),
  including the detail that the last segment is centre-padded with real audio from before it,
  not with silence.
- `tools/convert_htdemucs.py` converts the official checkpoints to GGUF. It needs `demucs` at
  conversion time only. `tools/dump_refs.py` and `stems-parity` are the parity harness.

## Authors

stems.cpp was written by [tinycrops](https://github.com/tinycrops), who did the port, the parity work
and the C ABI. It is maintained by betweentwomidnights alongside sa3.cpp, audiocraft.cpp and yuey.cpp
on the same ggml fork. tinycrops' original repo is at
[github.com/tinycrops/stems.cpp](https://github.com/tinycrops/stems.cpp).

## Credits

[demucs](https://github.com/facebookresearch/demucs) (Défossez et al., MIT, code and weights),
[ggml](https://github.com/ggml-org/ggml). The GGUF loader, WAV reader, HTTP session helpers
and build scripts come from betweentwomidnights' audiocraft.cpp and sa3.cpp (MIT).

# Distribution: GGUF naming, Hugging Face repos, model cards

How the stems.cpp weights are packaged and published, so anyone (and us, on a fresh machine) can
clone, download, build and run. This follows sa3.cpp's `docs/DISTRIBUTION.md`; the download
scripts, the server's model lookup and the model cards all follow this file.

## Naming

The GGUF convention (`ggml/docs/gguf.md`): `<BaseName>-<SizeLabel>-<Version>-<Encoding>.gguf`.

- **BaseName:** the model id the runtime reports and `stems-server` takes as `model`:
  `htdemucs`, `htdemucs_6s`, `htdemucs_ft`, `mel_band_roformer_kim`, `bs_roformer_viperx_317`,
  `uvr_denoise`, `uvr_denoise_lite`, `uvr_deecho_normal`, `uvr_deecho_aggressive`, `uvr_deecho_dereverb`,
  `bs_roformer_dereverb_room`, `mdx_net_kim_vocal_2`, `mdx23c_drumsep`.
  Underscores stay; `-` only separates fields.
- **SizeLabel:** the parameter class, rounded as sa3.cpp does: `0.xB` from 100M up, `NM` below.
  A bag of models is `<n>x<per-model>`, the convention's form for groups of experts.
- **Version:** `v1.0`. Bump it on any change to the weights or to the conversion.
- **Encoding:** `F32` or `F16`.

```
htdemucs-42M-v1.0-F32.gguf
htdemucs_6s-27M-v1.0-F32.gguf
htdemucs_ft-4x42M-v1.0-F32.gguf
mel_band_roformer_kim-0.2B-v1.0-F32.gguf
bs_roformer_viperx_317-0.2B-v1.0-F32.gguf
uvr_denoise-32M-v1.0-F32.gguf
uvr_deecho_dereverb-56M-v1.0-F32.gguf
mdx_net_kim_vocal_2-17M-v1.0-F32.gguf
mdx23c_drumsep-0.1B-v1.0-F32.gguf
```

Given a directory, `tools/convert_htdemucs.py`, `tools/convert_roformer.py`, `tools/convert_vr.py`
and `tools/convert_mdx.py` write these names themselves.

## Metadata

The converters keep `general.architecture` (`htdemucs`, `mel_band_roformer`, `bs_roformer`), which
the runtime dispatches on, and every `stems.*` / `roformer.*` key. Through `tools/gguf_meta.py`
they add:

| key | value |
|---|---|
| `general.basename` | the model id |
| `general.size_label` | `42M`, `27M`, `4x42M`, `0.2B` |
| `general.version` | `v1.0` |
| `general.license` | `mit` (HTDemucs, Kim), `other` (viperx) |
| `general.base_model.0.*` | upstream name, organization, repo URL and exact revision |
| `general.source.file` | upstream checkpoint file(s) |

Re-converting with these converters changes no tensor. The v1.0 F32 files were compared
tensor for tensor with the GGUFs every number in `docs/PARITY.md` was measured on, and all
match.

## Repositories

One repository per model family under [`thepatch`](https://huggingface.co/thepatch). Each holds
every encoding of its models, plus `README.md` (the card from `docs/model-cards/`), `LICENSE`,
`NOTICE` and `SHA256SUMS`.

| repo | models | license |
|---|---|---|
| [`thepatch/htdemucs-GGUF`](https://huggingface.co/thepatch/htdemucs-GGUF) | htdemucs, htdemucs_6s, htdemucs_ft | MIT (Meta) |
| [`thepatch/mel-band-roformer-kim-GGUF`](https://huggingface.co/thepatch/mel-band-roformer-kim-GGUF) | mel_band_roformer_kim | MIT (KimberleyJSN) |
| [`thepatch/bs-roformer-viperx-317-GGUF`](https://huggingface.co/thepatch/bs-roformer-viperx-317-GGUF) | bs_roformer_viperx_317 | none stated upstream |
| [`thepatch/uvr-vr-GGUF`](https://huggingface.co/thepatch/uvr-vr-GGUF) | uvr_denoise, uvr_denoise_lite, uvr_deecho_normal, uvr_deecho_aggressive, uvr_deecho_dereverb | none stated upstream |
| [`thepatch/bs-roformer-dereverb-room-GGUF`](https://huggingface.co/thepatch/bs-roformer-dereverb-room-GGUF) | bs_roformer_dereverb_room | GPL-3.0 (anvuew) |
| [`thepatch/mdx-net-kim-vocal-2-GGUF`](https://huggingface.co/thepatch/mdx-net-kim-vocal-2-GGUF) | mdx_net_kim_vocal_2 | none stated upstream |
| [`thepatch/mdx23c-drumsep-GGUF`](https://huggingface.co/thepatch/mdx23c-drumsep-GGUF) | mdx23c_drumsep | none stated upstream |

The three HTDemucs models share a repository: same source, architecture and license, the way
sa3.cpp's Stable Audio Open Small repo carries its finetunes. The RoFormers get one each, because
a repository's license metadata is per repository, and viperx's differs.

**viperx's BS-RoFormer has no license anywhere upstream.** UVR's model repo releases the
checkpoint without one. It is published with `license: other`, a NOTICE saying exactly that, and
an offer to remove it on request. Treat it as unlicensed.

**The UVR VR models have no license upstream either.** They share one repository (same source,
architecture and terms) and are published the same way as viperx: `license: other`, the caveat in
the card, and removal on request.

**anvuew's room dereverb is GPL-3.0.** Its repository carries `license: gpl-3.0`, the full GPL text
in `LICENSE`, and a card that records the GGUF conversion as a modification and names the upstream
checkpoint and the converter as its corresponding source. It is downloaded separately from the
runtime, like every other model.

**The two TFC-TDF models have no license upstream either.** Kim_Vocal_2 ships in UVR's model list
with no license, and DrumSep's original release (`github.com/jarredou/models`) is gone with no
license ever stated for the weights; its mirror's own code is AGPL-3.0, which says nothing about
them. Each is published like viperx: `license: other`, a NOTICE with the caveat, and removal on
request.

## Encodings

**F32 is the default and the reference.** Every number in `docs/PARITY.md` was measured on it.
**F16 is published per model, only where it was measured to hold up** against the same torch
references on CPU, CUDA and Vulkan (see each model card). HTDemucs' F16 stores only the
cross-transformer's matmul weights in half precision; its convolutions stay F32. A RoFormer's F16
stores every 2D matmul weight in half precision.

No quantized tiers yet. They would need the same per-model measurement first.

## Download

`./models.sh` (or `models.cmd` on Windows) fetches with curl, no Python:

```sh
./models.sh                                  # htdemucs, F32
./models.sh htdemucs_6s mel_band_roformer_kim
./models.sh --encoding f16 all               # F16 where published, F32 otherwise
./models.sh --dry-run all                    # print the URLs
```

`stems-server` finds a model by name in its models directory: canonical names first (F16, then
F32; the newest version wins), then the older `<name>-f16.gguf` / `<name>-f32.gguf`.

## Releasing

```sh
python tools/convert_htdemucs.py htdemucs_ft models/ [--f16]          # and the others
python tools/stage_hf_repos.py --gguf-dir models/ --out staging/ --f16 <models whose F16 passed>
hf upload thepatch/htdemucs-GGUF staging/htdemucs-GGUF . --repo-type model
```

`stage_hf_repos.py` fetches the demucs license from a pinned commit and verifies its checksum,
hard-links the GGUFs, writes `SHA256SUMS`, and refuses to replace a staged file that differs.
Review the staged trees before uploading.

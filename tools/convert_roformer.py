#!/usr/bin/env python3
"""Convert a Mel-Band RoFormer or BS-RoFormer checkpoint to GGUF.

    python tools/convert_roformer.py --preset kim models/            # -> mel_band_roformer_kim-0.2B-v1.0-F32.gguf
    python tools/convert_roformer.py --preset viperx --ckpt model_bs_roformer_ep_317_sdr_12.9755.ckpt \\
        --config model_bs_roformer_ep_317_sdr_12.9755.yaml models/   # -> bs_roformer_viperx_317-0.2B-v1.0-F32.gguf
    python tools/convert_roformer.py --arch bs_roformer --ckpt model.ckpt --config config.yaml \\
        --name bs_roformer_x models/bs_roformer_x-f32.gguf

Given a directory, the file gets its canonical name (docs/DISTRIBUTION.md); a file path is
used as is. --f16 stores the 2D matmul weights as F16.

Checkpoints are the state_dicts lucidrains' BS-RoFormer code produces, as trained with
ZFTurbo's Music-Source-Separation-Training (and Kim's Mel-Band RoFormer vocal model); the
config is that training YAML. Needs torch, numpy, pyyaml, librosa (Mel-Band only) and gguf at
conversion time; the C++ side never touches Python.

What the converter folds so the runtime is one code path for both architectures:
  * the band layout is written as plain index lists (roformer.band_sizes, roformer.band_freqs):
    contiguous ranges for BS-RoFormer, librosa's mel filterbank (binarised, exactly as the
    model's __init__ does) for Mel-Band;
  * the rotary embedding's frequencies are checked against theta 10000 and written as a theta;
  * tensors are renamed to the short names src/roformer.cpp documents.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_meta  # noqa: E402

PRESETS = {
    # KimberleyJSN/melbandroformer (MIT): vocals; the instrumental is the mix minus vocals.
    "kim": dict(
        arch="mel_band_roformer", name="mel_band_roformer_kim",
        ckpt="hf:KimberleyJSN/melbandroformer/MelBandRoformer.ckpt",
        source_url="https://huggingface.co/KimberleyJSN/melbandroformer", license="mit",
        source=dict(name="melbandroformer", organization="KimberleyJSN",
                    version="ac9b0614ab3cd7f77219e18ba494dfd93956c348", files=["MelBandRoformer.ckpt"]),
        config=dict(
            model=dict(dim=384, depth=6, stereo=True, num_stems=1, time_transformer_depth=1,
                       freq_transformer_depth=1, num_bands=60, dim_head=64, heads=8,
                       dim_freqs_in=1025, sample_rate=44100, stft_n_fft=2048,
                       stft_hop_length=441, stft_win_length=2048, stft_normalized=False,
                       mask_estimator_depth=2),
            training=dict(instruments=["vocals", "other"], target_instrument="vocals"),
            inference=dict(num_overlap=2, chunk_size=352800)),
        complement="instrumental", zero_dc=False),
    # viperx's BS-RoFormer ep_317 (vocals), as released in UVR's model repo. No license is
    # stated anywhere upstream, hence "other". The checkpoint and MSST's config YAML
    # (configs/viperx/) are local files: pass --ckpt and --config.
    "viperx": dict(
        arch="bs_roformer", name="bs_roformer_viperx_317", license="other",
        source_url="https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models",
        source=dict(name="BS-RoFormer ep_317 (viperx)", organization="TRvlvr/model_repo (UVR)",
                    version="sha256:5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa",
                    files=["model_bs_roformer_ep_317_sdr_12.9755.ckpt"]),
        complement="instrumental"),
}

ROPE_THETA = 10000.0


def load_ckpt(spec):
    import torch
    if spec.startswith("hf:"):
        from huggingface_hub import hf_hub_download
        repo, fname = spec[3:].rsplit("/", 1)
        spec = hf_hub_download(repo, fname)
    sd = torch.load(spec, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    return {k: v.detach().float().numpy() for k, v in sd.items()}


def mel_bands(m):
    """MelBandRoformer.__init__'s freqs_per_band, as index lists."""
    from librosa import filters
    fb = filters.mel(sr=m["sample_rate"], n_fft=m["stft_n_fft"], n_mels=m["num_bands"])
    fb[0][0] = 1.0
    fb[-1, -1] = 1.0
    mask = fb > 0
    assert mask.any(axis=0).all(), "every frequency must be covered by a band"
    return [np.nonzero(row)[0].tolist() for row in mask]


def bs_bands(m):
    bands, f = [], 0
    for n in m["freqs_per_bands"]:
        bands.append(list(range(f, f + n)))
        f += n
    assert f == m["stft_n_fft"] // 2 + 1, f"freqs_per_bands sum to {f}"
    return bands


def rename(k):
    """State-dict key -> GGUF tensor name; None to skip; raises on anything unexpected."""
    if k.endswith("rotary_embed.freqs"):
        return None
    if k == "final_norm.gamma":
        return "final_norm"
    m = re.fullmatch(r"band_split\.to_features\.(\d+)\.(0\.gamma|1\.weight|1\.bias)", k)
    if m:
        return f"band.{m[1]}." + {"0.gamma": "norm", "1.weight": "weight", "1.bias": "bias"}[m[2]]
    m = re.fullmatch(r"layers\.(\d+)\.([01])\.norm\.gamma", k)
    if m:
        return f"layer.{m[1]}.{'time' if m[2] == '0' else 'freq'}.norm"
    m = re.fullmatch(r"layers\.(\d+)\.([01])\.layers\.(\d+)\.([01])\.(.+)", k)
    if m:
        base = f"layer.{m[1]}.{'time' if m[2] == '0' else 'freq'}.{m[3]}."
        rest = {"norm.gamma": "attn_norm", "to_qkv.weight": "qkv", "to_gates.weight": "gate.weight",
                "to_gates.bias": "gate.bias", "to_out.0.weight": "out"} if m[4] == "0" else \
               {"net.0.gamma": "ff_norm", "net.1.weight": "ff1.weight", "net.1.bias": "ff1.bias",
                "net.4.weight": "ff2.weight", "net.4.bias": "ff2.bias"}
        if m[5] in rest:
            return base + rest[m[5]]
    m = re.fullmatch(r"mask_estimators\.(\d+)\.to_freqs\.(\d+)\.0\.(\d+)\.(weight|bias)", k)
    if m and int(m[3]) % 2 == 0:
        return f"mask.{m[1]}.{m[2]}.{int(m[3]) // 2}.{m[4]}"
    raise SystemExit(f"unexpected tensor {k} (an architecture variant this converter does not know)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="output file, or a directory to use the canonical file name")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--arch", choices=["mel_band_roformer", "bs_roformer"])
    ap.add_argument("--ckpt", help="checkpoint path, or hf:<repo>/<file>")
    ap.add_argument("--config", help="the training YAML")
    ap.add_argument("--name", help="model name the runtime reports, e.g. mel_band_roformer_kim")
    ap.add_argument("--complement", help="name for mix minus the target stem (single-stem models)")
    ap.add_argument("--source-url", default="")
    ap.add_argument("--license", default="")
    ap.add_argument("--f16", action="store_true", help="store 2D matmul weights as F16")
    args = ap.parse_args()

    p = dict(PRESETS[args.preset]) if args.preset else {}
    for k in ("arch", "ckpt", "name", "complement", "source_url", "license"):
        if getattr(args, k):
            p[k] = getattr(args, k)
    if args.config:
        import yaml
        with open(args.config) as f:
            p["config"] = yaml.load(f, Loader=yaml.FullLoader)
    for k in ("arch", "ckpt", "name", "config"):
        if k not in p:
            raise SystemExit(f"--{k} is required without a preset")

    cfg = p["config"]
    m, inf, tr = cfg["model"], cfg.get("inference", {}), cfg.get("training", {})
    channels = 2 if m.get("stereo", False) else 1
    if m.get("stft_win_length", m["stft_n_fft"]) != m["stft_n_fft"]:
        raise SystemExit("stft_win_length != stft_n_fft is not supported")
    for k in ("linear_transformer_depth", "use_torch_checkpoint", "skip_connection"):
        if m.get(k):
            raise SystemExit(f"unsupported model option {k}={m[k]}")
    bands = mel_bands(m) if p["arch"] == "mel_band_roformer" else bs_bands(m)

    n_stems = m.get("num_stems", 1)
    target = tr.get("target_instrument")
    if target:
        sources = [target]
    else:
        sources = list(tr["instruments"])[:n_stems]
    if len(sources) != n_stems:
        raise SystemExit(f"{n_stems} stems but sources {sources}")

    sd = load_ckpt(p["ckpt"])
    for k, v in sd.items():
        if k.endswith("rotary_embed.freqs"):
            dh = m.get("dim_head", 64)
            ref = 1.0 / (ROPE_THETA ** (np.arange(0, dh, 2, dtype=np.float32) / dh))
            if v.shape != ref.shape or not np.allclose(v, ref, rtol=1e-6):
                raise SystemExit(f"{k}: rotary frequencies are not theta={ROPE_THETA} over dim_head")

    w = GGUFWriter(None, p["arch"])   # the path depends on the parameter count, known at the end
    if p.get("source_url"):
        w.add_source_url(p["source_url"])
    w.add_string("stems.model", p["name"])
    w.add_array("stems.sources", sources)
    if p.get("complement") and n_stems == 1:
        w.add_string("stems.complement", p["complement"])
    w.add_uint32("stems.samplerate", m.get("sample_rate", 44100))
    w.add_uint32("stems.audio_channels", channels)
    u32 = lambda k, v: w.add_uint32("roformer." + k, int(v))
    u32("dim", m["dim"])
    u32("depth", m["depth"])
    u32("time_transformer_depth", m.get("time_transformer_depth", 2))
    u32("freq_transformer_depth", m.get("freq_transformer_depth", 2))
    u32("heads", m.get("heads", 8))
    u32("dim_head", m.get("dim_head", 64))
    # Counted from the checkpoint: Kim's copy of MLP builds depth + 1 linears, MSST's builds depth.
    mask_layers = 1 + max(int(re.fullmatch(r"mask\.\d+\.\d+\.(\d+)\.weight", n)[1])
                          for n in (rename(k) for k in sd) if n and n.startswith("mask.") and n.endswith(".weight"))
    u32("mask_layers", mask_layers)
    u32("stft_n_fft", m["stft_n_fft"])
    u32("stft_hop_length", m["stft_hop_length"])
    w.add_bool("roformer.stft_normalized", bool(m.get("stft_normalized", False)))
    # MSST's BSRoformer and MelBandRoformer zero the masked DC bin by default (zero_dc=True);
    # Kim's own MelBandRoformer, which the kim preset follows, has no such step.
    w.add_bool("roformer.zero_dc", bool(m.get("zero_dc", p.get("zero_dc", True))))
    w.add_float32("roformer.rope_theta", ROPE_THETA)
    u32("chunk_size", inf.get("chunk_size", cfg.get("audio", {}).get("chunk_size", 352800)))
    u32("num_overlap", inf.get("num_overlap", 2))
    w.add_array("roformer.band_sizes", [len(b) for b in bands])
    w.add_array("roformer.band_freqs", [f for b in bands for f in b])

    n = n_params = 0
    for k, v in sd.items():
        name = rename(k)
        if name is None:
            continue
        v = np.ascontiguousarray(v)
        if args.f16 and v.ndim == 2:
            w.add_tensor(name, v.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
        else:
            w.add_tensor(name, v.astype(np.float32))
        n += 1
        n_params += v.size
    want_bands = len(bands)
    got_bands = len({k.split(".")[1] for k in (rename(k) for k in sd) if k and k.startswith("band.")})
    if got_bands != want_bands:
        raise SystemExit(f"checkpoint has {got_bands} bands, config says {want_bands}")

    gguf_meta.add_general(w, p["name"], n_params, p.get("license") or "other")
    if p.get("source"):
        src = p["source"]
        gguf_meta.add_source(w, src["name"], src["organization"], p.get("source_url", ""),
                             src["version"], src.get("files"))

    out = args.out
    if os.path.isdir(out):
        out = os.path.join(out, gguf_meta.filename(p["name"], gguf_meta.size_label(n_params),
                                                   "F16" if args.f16 else "F32"))
    w.write_header_to_file(Path(out))
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {n} tensors, {n_params / 1e6:.1f}M params, {len(bands)} bands, sources {sources} -> {out}")


if __name__ == "__main__":
    sys.exit(main())

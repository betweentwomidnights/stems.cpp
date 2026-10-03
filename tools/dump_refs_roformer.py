#!/usr/bin/env python3
"""Dump PyTorch reference outputs of a RoFormer model for the parity tests.

    python tools/dump_refs_roformer.py kim tests/data/test.wav tests/refs/mel_band_roformer_kim
    python tools/dump_refs_roformer.py viperx_bs_317 tests/data/test.wav tests/refs/bs_roformer_viperx_317 \\
        --ckpt models/src/model_bs_roformer_ep_317_sdr_12.9755.ckpt

Writes, as .npy:
  seg_in.npy   [C, N]     the first chunk exactly as demix_track feeds it to the model
  seg_out.npy  [S, C, N]  the model's raw output on that chunk
  full.npy     [S, C, L]  demix_track over the whole file (the model's own stems, no complement)

The reference is the model code each checkpoint was published with, fetched at a pinned commit
into tests/refs/_code rather than vendored (Kim's repository carries no license). Chunking is
Kim's utils.demix_track for both; ZFTurbo's generic demix is the same algorithm. Runs on CPU in
float32, so the CUDA autocast in demix_track is a no-op, as it is in the original on CPU.
"""

import argparse
import importlib
import os
import sys
import urllib.request

import numpy as np
import soundfile as sf
import torch

KIM = ("KimberleyJensen/Mel-Band-Roformer-Vocal-Model", "25f44ffb55ee3c301281bba21b2d6d311cb69ae2")
MSST = ("ZFTurbo/Music-Source-Separation-Training", "84b1eac0887756b4f1a9d7a1ff49105939749ed2")

REFS = {
    "kim": dict(
        code=KIM, files=["models/mel_band_roformer/attend.py", "models/mel_band_roformer/mel_band_roformer.py"],
        config="configs/config_vocals_mel_band_roformer.yaml",
        module="models.mel_band_roformer.mel_band_roformer", cls="MelBandRoformer",
        ckpt="hf:KimberleyJSN/melbandroformer/MelBandRoformer.ckpt"),
    "viperx_bs_317": dict(
        code=MSST, files=["models/bs_roformer/attend.py", "models/bs_roformer/bs_roformer.py"],
        config="configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
        module="models.bs_roformer.bs_roformer", cls="BSRoformer", ckpt=None),
}


def fetch(code, files, root):
    repo, commit = code
    for f in files:
        dst = os.path.join(root, f)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{repo}/{commit}/{f}", dst)
        d = os.path.dirname(f)
        while d:   # empty package markers, so the upstream __init__ import chains are skipped
            init = os.path.join(root, d, "__init__.py")
            if not os.path.exists(init):
                open(init, "w").close()
            d = os.path.dirname(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", choices=sorted(REFS))
    ap.add_argument("wav")
    ap.add_argument("out")
    ap.add_argument("--ckpt", help="checkpoint path or hf:<repo>/<file> (overrides the preset)")
    args = ap.parse_args()
    ref = REFS[args.ref]

    base = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_code")
    root, kim_root = os.path.join(base, args.ref), os.path.join(base, "kim_utils")
    fetch(ref["code"], ref["files"] + [ref["config"]], root)
    fetch(KIM, ["utils.py"], kim_root)
    sys.path.insert(0, root)
    sys.path.insert(1, kim_root)
    import yaml
    from ml_collections import ConfigDict
    import utils

    with open(os.path.join(root, ref["config"])) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    if "chunk_size" not in config.inference:
        config.inference.chunk_size = config.audio.chunk_size
    config.model.flash_attn = False   # same math; torch's sdp_kernel context manager is gone
    cls = getattr(importlib.import_module(ref["module"]), ref["cls"])
    model = cls(**dict(config.model))

    ckpt = args.ckpt or ref["ckpt"]
    if not ckpt:
        raise SystemExit("--ckpt is required for " + args.ref)
    if ckpt.startswith("hf:"):
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(*ckpt[3:].rsplit("/", 1))
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd.get("state_dict", sd))
    model.eval()

    torch.set_num_threads(os.cpu_count())
    os.makedirs(args.out, exist_ok=True)
    save = lambda n, t: np.save(os.path.join(args.out, n + ".npy"),
                                (t.detach().numpy() if torch.is_tensor(t) else t).astype(np.float32))

    data, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    assert sr == config.model.get("sample_rate", 44100), sr
    mix = torch.from_numpy(data.T.copy())

    # The first chunk, as demix_track builds it (reflect border pad, then mix[:, 0:C]).
    C = config.inference.chunk_size
    border = C - C // config.inference.num_overlap
    padded = mix
    if mix.shape[1] > 2 * border and border > 0:
        padded = torch.nn.functional.pad(mix, (border, border), mode="reflect")
    seg = padded[:, :C]
    if seg.shape[1] < C:
        raise SystemExit("test audio shorter than one chunk")
    save("seg_in", seg)
    with torch.no_grad():
        y = model(seg[None])[0]
    if y.ndim == 2:
        y = y[None]
    save("seg_out", y)

    est, _ = utils.demix_track(config, model, mix, "cpu")
    full = np.stack([est[k] for k in est])
    save("full", full)
    print(f"wrote seg_in {tuple(seg.shape)}, seg_out {tuple(y.shape)}, full {full.shape} to {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert UVR 5.1 CascadedNet checkpoints to GGUF (no Python at inference).

python tools/convert_vr.py --preset denoise_lite models/
python tools/convert_vr.py --preset denoise models/
python tools/convert_vr.py --preset deecho_normal models/      # also deecho_aggressive, deecho_dereverb

Custom checkpoints require --ckpt, --params (UVR modelparams JSON), --name,
--primary and --secondary. Only the 5.1 network and ordinary stereo channels
are supported; older VR networks and channel transformations are rejected.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import torch
from gguf import GGUFWriter
from scipy.signal import firwin

import gguf_meta

UVR_REV = "a5f88453bfb2b38b05a965bcf67727243e0cbf19"
HF_REV = "6f4fc0c"
# (checkpoint, modelparams, SHA256, primary, secondary). The primary source is the one the
# network's mask selects, as in UVR's model_data.json; the secondary is its complement.
PRESETS = {
    "denoise_lite": ("UVR-DeNoise-Lite.pth", "1band_sr44100_hl1024", "0023492fe98c406817b5253965de19ede65d1c147db015a3a428f07602e99571", "noise", "denoised"),
    "denoise": ("UVR-DeNoise.pth", "4band_v3", "5addf43ece5bddd18da9f575a02d7ffdb32342414e6ad7ac8d1dd7a04138a628", "noise", "denoised"),
    "deecho_normal": ("UVR-De-Echo-Normal.pth", "4band_v3", "b849dd575643b075c257fb7a96c2ef5a79d7a5e7df74a2b319ad47118f1ee769", "no_echo", "echo"),
    "deecho_aggressive": ("UVR-De-Echo-Aggressive.pth", "4band_v3", "1bd1d79d9c5d1b17d20f96f8a9f8aff1b55a83014f70712446bf420c0188e0a0", "echo", "no_echo"),
    "deecho_dereverb": ("UVR-DeEcho-DeReverb.pth", "4band_v3", "e644028ec82865dc0fe082bc6fea85a43f7c71cfe375caee2da2d154aa661ee7", "no_reverb", "reverb"),
}


def band_config(params):
    """Validate the portable spectral contract, shared with the reference harness."""
    if any(params.get(k) for k in ("reverse", "mid_side", "mid_side_b2")):
        raise ValueError("legacy channel transformations are not supported")
    bands = [params["band"][str(i)] for i in range(1, len(params["band"]) + 1)]
    if not bands or sum(b["crop_stop"] - b["crop_start"] for b in bands) > params["bins"]:
        raise ValueError("invalid frequency band layout")
    if params["bins"] % 32:
        raise ValueError("bins must be divisible by 32")
    if not 32 <= params["bins"] <= 8192 or not 0 <= params["pre_filter_start"] < params["pre_filter_stop"] <= params["bins"]:
        raise ValueError("invalid bin count/prefilter")
    for b in bands:
        if b.get("convert_channels") or b.get("res_type") not in ("polyphase", "kaiser_fast", "sinc_best"):
            raise ValueError("unsupported band/channel configuration")
        if not (0 <= b["crop_start"] < b["crop_stop"] <= b["n_fft"] // 2 + 1):
            raise ValueError("invalid band crop")
        if b["sr"] <= 0 or b["hl"] <= 0 or b["n_fft"] < 2 or b["n_fft"] % 2:
            raise ValueError("invalid band STFT")
        for kind in ("hpf", "lpf"):
            start, stop = b.get(kind+"_start", -1), b.get(kind+"_stop", -1)
            if start < 0:
                continue
            if start < 1 or stop < 0 or max(start, stop) >= b["n_fft"]//2+1 or (start <= stop if kind == "hpf" else start >= stop):
                raise ValueError("invalid band filter")
    if bands[-1]["sr"] != params["sr"]:
        raise ValueError("highest band must run at the model sample rate")
    if any(b["sr"] * bands[-1]["hl"] != bands[-1]["sr"] * b["hl"] for b in bands):
        raise ValueError("band frames must have equal durations")
    return bands


def convert(sd):
    """Fold eval BatchNorm into Conv/Linear, preserve PyTorch tensor names otherwise."""
    out, used = {}, set()
    def take(k):
        used.add(k)
        return sd[k].detach().cpu().float().numpy()
    for k in sd:
        if k.startswith("aux_out.") or k in used:
            continue
        if k.endswith("conv.0.weight") or k.endswith("dense.0.weight"):
            base = k[:-len("0.weight")]
            weight = take(k)
            norm = base + "1."
            scale = take(norm + "weight") / np.sqrt(take(norm + "running_var") + np.float32(1e-5))
            bias = take(base + "0.bias") if base + "0.bias" in sd else np.zeros(len(scale), np.float32)
            out[base + "weight"] = weight * scale.reshape((-1,) + (1,) * (weight.ndim - 1))
            out[base + "bias"] = (bias - take(norm + "running_mean")) * scale + take(norm + "bias")
            take(norm + "num_batches_tracked")
        elif ".lstm." in k or k == "out.weight":
            out[k] = take(k)
    unknown = set(sd) - used - {k for k in sd if k.startswith("aux_out.")}
    if unknown:
        raise ValueError(f"unsupported VR checkpoint tensors: {sorted(unknown)[:8]}")
    return out


def validate_tensors(tensors, bins, nout, lstm):
    """The v1 tensor contract: reject architecture variants before writing a GGUF."""
    shapes = {}
    def cb(n, ni, no, k=3):
        shapes[n+".conv.weight"] = (no, ni, k, k)
        shapes[n+".conv.bias"] = (no,)
    def base(p, ni, no, freq, nl):
        cb(p+".enc1", ni, no)
        scales = (1, 2, 4, 6, 8)
        for i in range(1, 5):
            cb(p+f".enc{i+1}.conv1", no*scales[i-1], no*scales[i])
            cb(p+f".enc{i+1}.conv2", no*scales[i], no*scales[i])
        cb(p+".aspp.conv1.1", no*8, no*8, 1)
        cb(p+".aspp.conv2", no*8, no*8, 1)
        for i in range(3, 6):
            cb(p+f".aspp.conv{i}", no*8, no*8)
        cb(p+".aspp.bottleneck", no*40, no*8, 1)
        for k, ni, no_dec in ((4,no*14,no*6), (3,no*10,no*4), (2,no*6,no*2), (1,no*3+1,no)):
            cb(p+f".dec{k}.conv1", ni, no_dec)
        cb(p+".lstm_dec2.conv", no*2, 1, 1)
        for s in ("", "_reverse"):
            shapes[p+".lstm_dec2.lstm.weight_ih_l0"+s] = (nl*2, freq//2)
            shapes[p+".lstm_dec2.lstm.weight_hh_l0"+s] = (nl*2, nl//2)
            shapes[p+".lstm_dec2.lstm.bias_ih_l0"+s] = (nl*2,)
            shapes[p+".lstm_dec2.lstm.bias_hh_l0"+s] = (nl*2,)
        shapes[p+".lstm_dec2.dense.weight"] = (freq//2, nl)
        shapes[p+".lstm_dec2.dense.bias"] = (freq//2,)
    base("stg1_low_band_net.0", 2, nout//2, bins//2, lstm)
    cb("stg1_low_band_net.1", nout//2, nout//4, 1)
    base("stg1_high_band_net", 2, nout//4, bins//2, lstm//2)
    base("stg2_low_band_net.0", nout//4+2, nout, bins//2, lstm)
    cb("stg2_low_band_net.1", nout, nout//2, 1)
    base("stg2_high_band_net", nout//4+2, nout//2, bins//2, lstm//2)
    base("stg3_full_band_net", nout*3//4+2, nout, bins, lstm)
    shapes["out.weight"] = (2, nout, 1, 1)
    if set(tensors) != set(shapes):
        raise ValueError(f"VR tensor names differ: {sorted(set(tensors)^set(shapes))[:8]}")
    for k, shape in shapes.items():
        if tensors[k].shape != shape or not np.isfinite(tensors[k]).all():
            raise ValueError(f"invalid VR tensor {k}: expected {shape}, got {tensors[k].shape}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--preset", choices=PRESETS)
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--params", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--primary")
    ap.add_argument("--secondary")
    ap.add_argument("--license", default="other", help="exact weight license; no license assumed")
    ap.add_argument("--window", type=int, default=256)
    args = ap.parse_args()
    if args.window < 144 or args.window > 1024 or args.window % 16:
        ap.error("window must be a multiple of 16 between 144 and 1024")
    if args.preset:
        file, param_name, sha, default_primary, default_secondary = PRESETS[args.preset]
        ckpt = args.ckpt or Path("models/src") / file
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        if not ckpt.exists():
            urllib.request.urlretrieve(f"https://huggingface.co/seanghay/uvr_models/resolve/{HF_REV}/{file}", ckpt)
        if hashlib.sha256(ckpt.read_bytes()).hexdigest() != sha:
            raise ValueError("checkpoint SHA256 does not match the preset")
        if args.params:
            params = json.loads(args.params.read_text())
        else:
            url = f"https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/{UVR_REV}/lib_v5/vr_network/modelparams/{param_name}.json"
            params = json.load(urllib.request.urlopen(url))
        name = args.name or "uvr_" + args.preset
        primary, secondary = args.primary or default_primary, args.secondary or default_secondary
    else:
        if not all((args.ckpt, args.params, args.name, args.primary, args.secondary)):
            ap.error("custom VR models require --ckpt, --params, --name, --primary, --secondary")
        ckpt, name = args.ckpt, args.name
        params = json.loads(args.params.read_text())
        primary, secondary = args.primary, args.secondary
    bands = band_config(params)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    tensors = convert(sd)
    # Infer capacity from weights, never from file size. The native loader validates
    # every layer against the metadata before constructing its compute graph.
    nout = int(sd["stg3_full_band_net.enc1.conv.0.weight"].shape[0])
    lstm = int(sd["stg3_full_band_net.lstm_dec2.lstm.weight_hh_l0"].shape[1]) * 2
    if not 4 <= nout <= 256 or nout % 4 or not 4 <= lstm <= 512 or lstm % 4:
        raise ValueError("unsupported network capacity")
    if sd["stg3_full_band_net.lstm_dec2.lstm.weight_ih_l0"].shape[1] != params["bins"] // 2:
        raise ValueError("checkpoint frequency bins do not match modelparams")
    validate_tensors(tensors, params["bins"], nout, lstm)
    w = GGUFWriter(None, "vr_cascaded")
    w.add_string("stems.model", name)
    w.add_array("stems.sources", [primary, secondary])
    w.add_uint32("stems.samplerate", params["sr"])
    w.add_uint32("stems.audio_channels", 2)
    for k, v in dict(schema_version=1, bins=params["bins"], nout=nout, nout_lstm=lstm,
                     window=args.window, offset=64, band_count=len(bands),
                     pre_filter_start=params["pre_filter_start"], pre_filter_stop=params["pre_filter_stop"]).items():
        w.add_uint32("vr." + k, v)
    w.add_string("vr.resampling", "polyphase")
    for i, b in enumerate(bands):
        for k in ("sr", "hl", "n_fft", "crop_start", "crop_stop", "hpf_start", "hpf_stop", "lpf_start", "lpf_stop"):
            w.add_int32(f"vr.band.{i}.{k}", b.get(k, -1))
    # scipy.signal.resample_poly's default Kaiser FIR, including upsampling gain.
    # Store coefficients, not a dependency on scipy at inference time.
    from math import gcd
    rates = {b["sr"] for b in bands}
    for src in rates:
        for dst in rates:
            if src == dst:
                continue
            g = gcd(src, dst)
            up, down = dst // g, src // g
            m = max(up, down)
            h = firwin(20 * m + 1, 1.0 / m, window=("kaiser", 5.0)).astype(np.float32) * up
            tensors[f"resample.{src}.{dst}"] = h
    n_params = sum(v.numel() for k, v in sd.items() if not k.endswith(("num_batches_tracked", "running_mean", "running_var")))
    for k, a in tensors.items():
        # Channel biases broadcast over [time, freq, channel]. Dense biases are dim0.
        if k.endswith("conv.bias"):
            a = a.reshape(-1, 1, 1)
        w.add_tensor(k, np.ascontiguousarray(a, dtype=np.float32))
    gguf_meta.add_general(w, name, n_params, args.license)
    w.add_source_url("https://huggingface.co/seanghay/uvr_models")
    gguf_meta.add_source(w, ckpt.name, "UVR", "https://huggingface.co/seanghay/uvr_models",
                         "sha256:" + hashlib.sha256(ckpt.read_bytes()).hexdigest(), [ckpt.name])
    out = args.out / gguf_meta.filename(name, gguf_meta.size_label(n_params), "F32") if args.out.is_dir() else args.out
    w.write_header_to_file(out)
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{name}: bins={params['bins']}, capacity={nout}/{lstm}, {len(bands)} bands -> {out}")


if __name__ == "__main__":
    main()

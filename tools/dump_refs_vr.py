#!/usr/bin/env python3
"""Pinned UVR network references and an explicit portable VR spectral reference.

python tools/dump_refs_vr.py denoise_lite input.wav tests/refs/vr_lite
python tools/dump_refs_vr.py denoise input.wav tests/refs/vr_full

The network is upstream UVR, unchanged. DSP is defined here: zero STFT borders,
polyphase resampling for analysis AND synthesis, no aggressiveness/TTA/artifact
merging/high-end mirroring. UVR's Windows synthesis defaults to sinc_fastest;
this portable mode deliberately uses scipy polyphase on every platform.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import types
import urllib.request

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from convert_vr import PRESETS, UVR_REV, band_config


def network(ckpt, params, cache):
    root = cache / "vr_reference"
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").touch()
    for file in ("nets_new.py", "layers_new.py"):
        path = root / file
        if not path.exists():
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/{UVR_REV}/lib_v5/vr_network/{file}", path)
    # Upstream layers needs only crop_center from the GUI's large spec_utils module.
    spec = types.ModuleType("lib_v5.spec_utils")
    def crop_center(a, b):
        if a.shape[-1] < b.shape[-1]:
            raise ValueError("negative crop")
        left = (a.shape[-1] - b.shape[-1]) // 2
        return a[..., left:left+b.shape[-1]]
    spec.crop_center = crop_center
    package = types.ModuleType("lib_v5")
    package.spec_utils = spec
    sys.modules["lib_v5"] = package
    sys.modules["lib_v5.spec_utils"] = spec
    sys.path.insert(0, str(cache))
    net = importlib.import_module("vr_reference.nets_new")
    sd = torch.load(ckpt, weights_only=True, map_location="cpu")
    nout = sd["stg3_full_band_net.enc1.conv.0.weight"].shape[0]
    nl = sd["stg3_full_band_net.lstm_dec2.lstm.weight_hh_l0"].shape[1]*2
    model = net.CascadedNet(params["bins"]*2, nout=nout, nout_lstm=nl)
    model.load_state_dict(sd, strict=True)
    return model.eval()


def resample(x, src, dst):
    if src == dst:
        return x
    from math import gcd
    g = gcd(src, dst)
    return resample_poly(x, dst//g, src//g, axis=-1).astype(np.float32)


def lp(F, start, stop):
    if start < 0:
        return np.ones(F)
    return np.concatenate((np.ones(start-1), np.linspace(1, 0, stop-start+1), np.zeros(F-stop)))


def hp(F, start, stop):
    if start <= 0:
        return np.ones(F)
    stop -= 1
    return np.concatenate((np.zeros(stop+1), np.linspace(0, 1, 1+start-stop), np.ones(F-start-2)))


def spectrum(wave, params):
    bands = band_config(params)
    specs = [None]*len(bands)
    rate = params["sr"]
    for i in range(len(bands)-1, -1, -1):
        b = bands[i]
        wave = resample(wave, rate, b["sr"])
        rate = b["sr"]
        specs[i] = librosa.stft(wave, n_fft=b["n_fft"], hop_length=b["hl"], pad_mode="constant")
    T = min(s.shape[-1] for s in specs)
    out = np.zeros((2, params["bins"]+1, T), np.complex64)
    off = 0
    for b, s in zip(bands, specs):
        h = b["crop_stop"]-b["crop_start"]
        out[:, off:off+h] = s[:, b["crop_start"]:b["crop_stop"], :T]
        off += h
    out *= lp(out.shape[1], params["pre_filter_start"], params["pre_filter_stop"])[:, None]
    return out


def synthesize(spec, params, length):
    bands = band_config(params)
    wave, off = None, 0
    for i, b in enumerate(bands):
        F = b["n_fft"]//2+1
        part = np.zeros((2, F, spec.shape[-1]), np.complex64)
        h = b["crop_stop"]-b["crop_start"]
        part[:, b["crop_start"]:b["crop_stop"]] = spec[:, off:off+h]
        off += h
        part *= hp(F, b.get("hpf_start", -1), b.get("hpf_stop", -1))[:, None]
        if i+1 < len(bands):
            part *= lp(F, b.get("lpf_start", -1), b.get("lpf_stop", -1))[:, None]
        band_wave = librosa.istft(part, hop_length=b["hl"])
        wave = band_wave if wave is None else wave+band_wave
        if i+1 < len(bands):
            wave = resample(wave, b["sr"], bands[i+1]["sr"])
    return librosa.util.fix_length(wave, size=length)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("preset", choices=PRESETS)
    ap.add_argument("wav", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--window", type=int, default=256)
    args = ap.parse_args()
    file, param_name, _ = PRESETS[args.preset]
    params = json.load(urllib.request.urlopen(f"https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/{UVR_REV}/lib_v5/vr_network/modelparams/{param_name}.json"))
    torch.set_num_threads(4)
    model = network(args.ckpt or Path("models/src")/file, params, args.out.parent/"_vr_code")
    wave, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    if sr != params["sr"] or wave.shape[1] != 2:
        raise ValueError("reference WAV must be stereo at the model sample rate")
    wave = wave.T.copy()
    spec = spectrum(wave, params)
    mag = np.abs(spec)
    roi = args.window-2*model.offset
    if roi <= 0:
        raise ValueError("window too small")
    T = mag.shape[-1]
    padded = np.pad(mag, ((0,0), (0,0), (model.offset, roi-T%roi+model.offset)))
    peak = padded.max()
    if peak:
        padded /= peak
    masks = []
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out/"mask_in.npy", padded[..., :args.window])
    with torch.inference_mode():
        for i in range(T//roi+1):
            patch = torch.from_numpy(padded[..., i*roi:i*roi+args.window][None].copy())
            mask = model.predict_mask(patch).numpy()[0]
            if i == 0:
                np.save(args.out/"mask_out.npy", mask)
            masks.append(mask)
    mask = np.concatenate(masks, axis=-1)[..., :T]
    full = np.stack([synthesize(spec*mask, params, wave.shape[-1]),
                     synthesize(spec*(1-mask), params, wave.shape[-1])])
    np.save(args.out/"full.npy", full.astype(np.float32))
    # Silence and atypical input lengths are exercised separately by the native DSP tests.
    print(f"{args.preset}: {T} frames, {len(masks)} windows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

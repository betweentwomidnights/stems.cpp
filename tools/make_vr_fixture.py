#!/usr/bin/env python3
"""Create the deterministic noisy stereo fixture used for VR parity.

python tools/make_vr_fixture.py tests/data/vr-noisy.wav
python tools/make_vr_fixture.py tests/data/vr-music.wav --source tests/data/test.wav
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--source", type=Path, help="take a 4.003 s excerpt from a stereo 44.1 kHz WAV")
    args = ap.parse_args()
    if args.source:
        wave, sr = sf.read(args.source, dtype="float32", always_2d=True)
        if sr != 44100 or wave.shape[1] != 2:
            ap.error("source must be stereo at 44100 Hz")
        wave = wave[:4*sr+127]
    else:
        sr = 44100
        n = sr*3+113
        t = np.arange(n)/sr
        rng = np.random.default_rng(42)
        wave = np.stack([.1*np.sin(2*np.pi*220*t)+.04*rng.normal(size=n),
                         .07*np.sin(2*np.pi*330*t)+.03*rng.normal(size=n)], axis=-1).astype("float32")
        wave[2000:2060, 0] += .3*np.hanning(60)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, wave, sr, subtype="FLOAT")
    print(f"{args.out}: {len(wave)/sr:.3f} s stereo at {sr} Hz")


if __name__ == "__main__":
    main()

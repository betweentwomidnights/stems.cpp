#!/usr/bin/env python3
"""Reference outputs of the TFC-TDF models for stems-parity.

python tools/dump_refs_mdx.py kim_vocal_2 tests/data/test.wav tests/refs/mdx_net_kim_vocal_2
python tools/dump_refs_mdx.py drumsep tests/data/test.wav tests/refs/mdx23c_drumsep

Writes, as .npy:
  seg_in.npy   [2, N]     the first chunk exactly as the chunking feeds it to the model
  seg_out.npy  [S, 2, N]  the model's raw output on that chunk (waveform, after the iSTFT)
  full.npy     [S', 2, L] the whole file: every source the GGUF lists, complement included

kim_vocal_2 runs the unchanged ONNX file in onnxruntime (UVR's own path at the default 256-frame
segment) with UVR's STFT class, fetched at the pinned UVR revision. UVR's SeperateMDX.demix and
run_model are transcribed below at the GUI defaults: "Default" overlap, no denoise, no pitch
shift, volume compensation from model_data.json, secondary stem = mix - compensated primary.
They live in separate.py, which cannot be imported without the GUI.

drumsep runs ZFTurbo's mdx23c_tfc_tdf_v3.py and its generic utils.model_utils.demix, fetched at
the pinned commit, on CPU in float32 (demix's CUDA autocast is a no-op on CPU).

Needs numpy, soundfile, torch, plus onnx and onnxruntime (kim_vocal_2) or pyyaml and
ml_collections (drumsep). Checkpoints come from models/src/ as tools/convert_mdx.py leaves them.
"""
import argparse
import importlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_mdx import PRESETS, UVR_REV, fetch  # noqa: E402

MSST = ("ZFTurbo/Music-Source-Separation-Training", "84b1eac0887756b4f1a9d7a1ff49105939749ed2")


def fetch_code(repo, commit, files, root):
    for f in files:
        dst = root / f
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{repo}/{commit}/{f}", dst)
        d = dst.parent
        while d != root:   # empty package markers, so upstream __init__ import chains are skipped
            (d / "__init__.py").touch()
            d = d.parent


class UVRMDX:
    """SeperateMDX (separate.py) at UVR's defaults, for an ONNX MDX-Net model."""

    def __init__(self, onnx_path, p, code):
        import onnxruntime as ort
        fetch_code("Anjok07/ultimatevocalremovergui", UVR_REV, ["lib_v5/tfc_tdf_v3.py"], code)
        sys.path.insert(0, str(code))
        STFT = importlib.import_module("lib_v5.tfc_tdf_v3").STFT
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.model_run = lambda spek: sess.run(None, {"input": spek.cpu().numpy()})[0]
        self.n_fft, self.hop, self.dim_f = p["n_fft"], p["hop"], p["dim_f"]
        self.compensate = p["compensate"]
        self.mdx_segment_size = p["dim_t"]                 # UVR default 256 == the model's dim_t
        # initialize_model_settings
        self.trim = self.n_fft // 2
        self.chunk_size = self.hop * (self.mdx_segment_size - 1)
        self.stft = STFT(self.n_fft, self.hop, self.dim_f, "cpu")
        self.adjust = 1

    def run_model(self, mix):
        spek = self.stft(mix) * self.adjust
        spek[:, :, :3, :] *= 0
        spec_pred = self.model_run(spek)
        return self.stft.inverse(torch.tensor(spec_pred)).cpu().detach().numpy()

    def first_chunk(self, mix):
        gen_size = self.chunk_size - 2 * self.trim
        pad = gen_size + self.trim - (mix.shape[-1] % gen_size)
        mixture = np.concatenate((np.zeros((2, self.trim), dtype="float32"), mix,
                                  np.zeros((2, pad), dtype="float32")), 1)
        return mixture[:, :self.chunk_size]

    def demix(self, mix):
        chunk_size = self.chunk_size
        gen_size = chunk_size - 2 * self.trim
        pad = gen_size + self.trim - (mix.shape[-1] % gen_size)
        mixture = np.concatenate((np.zeros((2, self.trim), dtype="float32"), mix,
                                  np.zeros((2, pad), dtype="float32")), 1)
        step = self.chunk_size - self.n_fft                # overlap == DEFAULT
        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        for i in range(0, mixture.shape[-1], step):
            start, end = i, min(i + chunk_size, mixture.shape[-1])
            chunk_size_actual = end - start
            window = np.tile(np.hanning(chunk_size_actual)[None, None, :], (1, 2, 1))
            mix_part_ = mixture[:, start:end]
            if end != i + chunk_size:
                mix_part_ = np.concatenate((mix_part_, np.zeros((2, (i + chunk_size) - end), dtype="float32")), axis=-1)
            mix_part = torch.tensor(np.array([mix_part_]), dtype=torch.float32)
            with torch.no_grad():
                tar_waves = self.run_model(mix_part)
            tar_waves[..., :chunk_size_actual] *= window
            divider[..., start:end] += window
            result[..., start:end] += tar_waves[..., :end - start]
        with np.errstate(invalid="ignore", divide="ignore"):   # 0/0 only inside the trimmed edges
            tar_waves = result / divider
        tar_waves_ = np.vstack([tar_waves])[:, :, self.trim:-self.trim]
        tar_waves = np.concatenate(tar_waves_, axis=-1)[:, :mix.shape[-1]]
        return tar_waves * self.compensate


def kim(args, p, mix):
    m = UVRMDX(args.src or Path("models/src") / p["file"], p, args.out.parent / "_mdx_code" / "uvr")
    seg = m.first_chunk(mix)
    with torch.no_grad():
        seg_out = m.run_model(torch.tensor(seg[None]))
    vocals = m.demix(mix)
    return seg, seg_out, np.stack([vocals, mix - vocals])


def drumsep(args, p, mix):
    import yaml
    from ml_collections import ConfigDict
    code = args.out.parent / "_mdx_code" / "msst"
    fetch_code(*MSST, ["models/mdx23c_tfc_tdf_v3.py", "utils/model_utils.py"], code)
    sys.path.insert(0, str(code))
    src = args.src or Path("models/src") / p["file"]
    fetch(p["url"], src, p["sha256"])
    cfg_path = src.with_name(src.name + ".yaml")
    fetch(p["config_url"], cfg_path, p["config_sha256"])
    config = ConfigDict(yaml.load(cfg_path.read_text(), Loader=yaml.FullLoader))
    model = importlib.import_module("models.mdx23c_tfc_tdf_v3").TFC_TDF_net(config)
    sd = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(sd.get("state_dict", sd))
    model.eval()
    utils = importlib.import_module("utils.model_utils")

    x = torch.from_numpy(mix)
    C = config.audio.chunk_size
    border = C - C // config.inference.num_overlap
    padded = x
    if x.shape[1] > 2 * border and border > 0:
        padded = torch.nn.functional.pad(x, (border, border), mode="reflect")
    seg = padded[:, :C]
    if seg.shape[1] < C:
        raise SystemExit("test audio shorter than one chunk")
    with torch.no_grad():
        seg_out = model(seg[None])[0]
    est = utils.demix(config, model, mix, torch.device("cpu"), "mdx23c")
    return seg.numpy(), seg_out.numpy(), np.stack([est[k] for k in config.training.instruments])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=PRESETS)
    ap.add_argument("wav", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--src", type=Path, help="local checkpoint (default: models/src/<preset file>)")
    args = ap.parse_args()
    p = PRESETS[args.preset]
    torch.set_num_threads(4)
    data, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    if sr != 44100 or data.shape[1] != 2:
        raise SystemExit("reference WAV must be stereo at 44100 Hz")
    mix = np.ascontiguousarray(data.T)
    seg, seg_out, full = (kim if p["variant"] == "mdx_net" else drumsep)(args, p, mix)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, a in (("seg_in", seg), ("seg_out", seg_out), ("full", full)):
        np.save(args.out / f"{name}.npy", np.ascontiguousarray(a, dtype=np.float32))
    print(f"{args.preset}: seg_in {seg.shape}, seg_out {seg_out.shape}, full {full.shape} -> {args.out}")


if __name__ == "__main__":
    main()

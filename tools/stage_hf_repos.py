#!/usr/bin/env python3
"""Stage the stems.cpp Hugging Face model repositories from a directory of converted GGUFs.

    python tools/stage_hf_repos.py --gguf-dir models/ --out staging/

Maintainer tooling, not a downloader and not an uploader (cf. sa3.cpp's
tools/stage_training_base_repos.py). For each repository in docs/DISTRIBUTION.md it builds
<out>/<repo>/ with the model card as README.md, LICENSE, NOTICE, SHA256SUMS and the GGUFs
(hard-linked when possible, copied otherwise). The demucs license is fetched from a pinned
upstream commit and checksum-verified. Existing files with different contents are an error, not
silently replaced. Review the staged trees, then upload with the standard tooling:

    hf upload thepatch/htdemucs-GGUF staging/htdemucs-GGUF . --repo-type model
"""
import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "v1.0"

DEMUCS_LICENSE = dict(
    url="https://raw.githubusercontent.com/facebookresearch/demucs/3b8430c12242bbbba48769eed6da5190c6ff3c2d/LICENSE",
    sha256="cf9b17822d1fcd4ff32ccbe14183386fb3adf6f2ff92dc184130823f7fc28173")

# Kim's checkpoint is declared MIT in its Hugging Face metadata, which ships no LICENSE file and
# names no copyright holder, so the standard MIT text is attributed to the uploading account.
KIM_LICENSE = """MIT License

Copyright (c) KimberleyJSN (https://huggingface.co/KimberleyJSN)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

MODIFICATION = ("These files are a re-encoding of the upstream weights as GGUF for stems.cpp: tensors are "
                "renamed, some are reshaped or split (e.g. packed attention projections into q/k/v), constants "
                "are folded into metadata, and any F16 files store matmul weights in half precision. The model "
                "architecture and its training are unchanged.")

REPOS = {
    "htdemucs-GGUF": dict(
        models={"htdemucs": "42M", "htdemucs_6s": "27M", "htdemucs_ft": "4x42M"},
        license=("url", DEMUCS_LICENSE),
        notice="HTDemucs (Hybrid Transformer Demucs) by Meta Platforms, Inc. and affiliates, from "
               "https://github.com/facebookresearch/demucs (demucs 4.0.1), released under the MIT License "
               "included as LICENSE. Checkpoints: htdemucs 955717e8-8726e21a.th; htdemucs_6s "
               "5c90dfd2-34c22ccb.th; htdemucs_ft f7e0c4bc-ba3fe64a.th, d12395a8-e57c48e6.th, "
               "92cfc3b6-ef3bcb9c.th, 04573f0d-f3cf25b2.th."),
    "mel-band-roformer-kim-GGUF": dict(
        models={"mel_band_roformer_kim": "0.2B"},
        license=("text", KIM_LICENSE),
        notice="Mel-Band RoFormer vocal model by Kimberley Jensen, from "
               "https://huggingface.co/KimberleyJSN/melbandroformer (MelBandRoformer.ckpt, revision "
               "ac9b0614ab3cd7f77219e18ba494dfd93956c348). That repository declares the MIT License in its "
               "metadata but ships no license file and names no copyright holder, so LICENSE carries the "
               "standard MIT text attributed to the uploading account."),
    "bs-roformer-viperx-317-GGUF": dict(
        models={"bs_roformer_viperx_317": "0.2B"},
        license=None,
        notice="BS-RoFormer vocal model trained by viperx, as released in Ultimate Vocal Remover's model "
               "repository (https://github.com/TRvlvr/model_repo, release all_public_uvr_models, file "
               "model_bs_roformer_ep_317_sdr_12.9755.ckpt, sha256 "
               "5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa). Its configuration is from "
               "ZFTurbo's Music-Source-Separation-Training (MIT, configs/viperx/). NO LICENSE IS STATED for "
               "these weights anywhere upstream. They are redistributed here in GGUF form with that caveat; "
               "the rights holder can ask for their removal. Do not assume any license."),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def put_bytes(dst, data):
    if dst.exists():
        if dst.read_bytes() != data:
            raise SystemExit(f"{dst} exists with different contents; refusing to replace it")
        return
    dst.write_bytes(data)


def put_file(dst, src):
    if dst.exists():
        if sha256(dst) != sha256(src):
            raise SystemExit(f"{dst} exists with different contents; refusing to replace it")
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf-dir", required=True, help="directory with the converted GGUFs")
    ap.add_argument("--out", required=True, help="staging directory, one subdirectory per repo")
    ap.add_argument("--encodings", default="F32,F16",
                    help="encodings to stage when present, in order (default F32,F16)")
    ap.add_argument("--f16", default="",
                    help="comma list of models whose F16 is published; F16 files of others are skipped")
    args = ap.parse_args()
    src_dir, out = Path(args.gguf_dir), Path(args.out)
    f16_ok = set(filter(None, args.f16.split(",")))
    cards = ROOT / "docs" / "model-cards"
    license_cache = {}

    for repo, spec in REPOS.items():
        d = out / repo
        d.mkdir(parents=True, exist_ok=True)
        files = []
        for model, label in spec["models"].items():
            for enc in args.encodings.split(","):
                if enc == "F16" and model not in f16_ok:
                    continue
                name = f"{model}-{label}-{VERSION}-{enc}.gguf"
                if not (src_dir / name).exists():
                    if enc == "F32":
                        raise SystemExit(f"missing {src_dir / name}")
                    continue
                put_file(d / name, src_dir / name)
                files.append(name)
        card = cards / f"{repo}.md"
        if not card.exists():
            raise SystemExit(f"missing model card {card}")
        put_bytes(d / "README.md", card.read_bytes())
        if spec["license"]:
            kind, value = spec["license"]
            if kind == "url":
                if value["url"] not in license_cache:
                    data = urllib.request.urlopen(value["url"]).read()
                    if hashlib.sha256(data).hexdigest() != value["sha256"]:
                        raise SystemExit(f"license checksum mismatch for {value['url']}")
                    license_cache[value["url"]] = data
                put_bytes(d / "LICENSE", license_cache[value["url"]])
            else:
                put_bytes(d / "LICENSE", value.encode())
        put_bytes(d / "NOTICE", (spec["notice"] + "\n\n" + MODIFICATION + "\n").encode())
        sums = "".join(f"{sha256(d / f)}  {f}\n" for f in sorted(files))
        if (d / "SHA256SUMS").exists():
            (d / "SHA256SUMS").unlink()
        (d / "SHA256SUMS").write_text(sums, newline="\n")
        total = sum((d / f).stat().st_size for f in files)
        print(f"{repo}: {len(files)} GGUFs, {total / 2**20:.0f} MiB -> {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

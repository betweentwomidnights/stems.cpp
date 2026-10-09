"""Check which file stems-server picks for each model name, with empty stand-in files, so no
model download is needed. Reads the choice from `stems-server --props --models-dir DIR`.

    python tests/model_resolution_test.py <stems-server>
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


binary = sys.argv[1]
failures = 0


def resolved(files):
    with tempfile.TemporaryDirectory() as d:
        for name in files:
            (Path(d) / name).write_bytes(b"")
        out = subprocess.check_output([binary, "--props", "--models-dir", d], text=True, timeout=60)
    props = json.loads(out)
    assert props["models"]["directory"] == d
    return props["models"]["resolved"]


def case(what, files, expect):
    global failures
    got = resolved(files)
    wrong = {k: (got.get(k), v) for k, v in expect.items() if got.get(k) != v}
    print(f"  {what:<58} {'ok' if not wrong else 'FAIL ' + repr(wrong)}")
    failures += bool(wrong)


case("empty directory resolves nothing", [],
     {"htdemucs": None, "htdemucs_6s": None, "htdemucs_ft": None,
      "mel_band_roformer_kim": None, "bs_roformer_viperx_317": None})
case("canonical F32", ["htdemucs-42M-v1.0-F32.gguf"],
     {"htdemucs": "htdemucs-42M-v1.0-F32.gguf", "htdemucs_6s": None})
case("F16 preferred over F32", ["htdemucs-42M-v1.0-F32.gguf", "htdemucs-42M-v1.0-F16.gguf"],
     {"htdemucs": "htdemucs-42M-v1.0-F16.gguf"})
case("newest version of an encoding wins", ["htdemucs-42M-v1.0-F32.gguf", "htdemucs-42M-v1.1-F32.gguf"],
     {"htdemucs": "htdemucs-42M-v1.1-F32.gguf"})
case("htdemucs does not match htdemucs_6s or _ft files",
     ["htdemucs_6s-27M-v1.0-F32.gguf", "htdemucs_ft-4x42M-v1.0-F16.gguf"],
     {"htdemucs": None, "htdemucs_6s": "htdemucs_6s-27M-v1.0-F32.gguf",
      "htdemucs_ft": "htdemucs_ft-4x42M-v1.0-F16.gguf"})
case("legacy names still resolve", ["mel_band_roformer_kim-f32.gguf", "htdemucs.gguf"],
     {"mel_band_roformer_kim": "mel_band_roformer_kim-f32.gguf", "htdemucs": "htdemucs.gguf"})
case("legacy -f16 before -f32 before bare", ["htdemucs-f32.gguf", "htdemucs-f16.gguf", "htdemucs.gguf"],
     {"htdemucs": "htdemucs-f16.gguf"})
case("canonical beats legacy", ["htdemucs-f16.gguf", "htdemucs-42M-v1.0-F32.gguf"],
     {"htdemucs": "htdemucs-42M-v1.0-F32.gguf"})
case("malformed names are ignored", ["htdemucs-42M-F32.gguf", "htdemucs-42M-1.0-F32.gguf", "htdemucs-a-b-c-d.gguf"],
     {"htdemucs": None})
case("viperx (F32 only)", ["bs_roformer_viperx_317-0.2B-v1.0-F32.gguf"],
     {"bs_roformer_viperx_317": "bs_roformer_viperx_317-0.2B-v1.0-F32.gguf"})
case("VR full and Lite resolve independently",
     ["uvr_denoise-32M-v1.0-F32.gguf", "uvr_denoise_lite-4M-v1.0-F32.gguf"],
     {"uvr_denoise": "uvr_denoise-32M-v1.0-F32.gguf",
      "uvr_denoise_lite": "uvr_denoise_lite-4M-v1.0-F32.gguf"})
case("VR De-Echo variants resolve independently",
     ["uvr_deecho_normal-32M-v1.0-F32.gguf", "uvr_deecho_aggressive-32M-v1.0-F32.gguf",
      "uvr_deecho_dereverb-56M-v1.0-F32.gguf"],
     {"uvr_deecho_normal": "uvr_deecho_normal-32M-v1.0-F32.gguf",
      "uvr_deecho_aggressive": "uvr_deecho_aggressive-32M-v1.0-F32.gguf",
      "uvr_deecho_dereverb": "uvr_deecho_dereverb-56M-v1.0-F32.gguf", "uvr_denoise": None})
case("room dereverb RoFormer", ["bs_roformer_dereverb_room-29M-v1.0-F32.gguf"],
     {"bs_roformer_dereverb_room": "bs_roformer_dereverb_room-29M-v1.0-F32.gguf"})

print(f"FAIL: {failures} case(s)" if failures else "PASS")
sys.exit(1 if failures else 0)

"""Verify an extracted stems.cpp Linux package the way a host uses it, without models.

The tools must start from the package alone, and libstems.so, loaded by absolute path
from a program outside the package (stems-abi-test, as gary4juce loads it), must find its
ggml libraries and register its backends from its own folder.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    if not __debug__:
        raise RuntimeError("Package verification requires assertions; remove PYTHONOPTIMIZE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("version")
    parser.add_argument("--expect-platform", required=True)
    parser.add_argument("--abi-test", type=Path, required=True, help="the build's stems-abi-test")
    parser.add_argument("--require-backend", choices=["cuda", "vulkan"])
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    version = args.version.removeprefix("v")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("STEMS_", "GGML_")) and k != "LD_LIBRARY_PATH"}
    with tempfile.TemporaryDirectory(prefix="stems-package-probe-") as scratch:
        models = Path(scratch) / "models"
        models.mkdir()

        def output(program, *flags):
            result = subprocess.run([str(program), *flags], cwd=scratch, env=env,
                                    capture_output=True, text=True, timeout=120)
            assert result.returncode == 0, (program, flags, result.stdout, result.stderr)
            return result.stdout.strip()

        for tool in ("stems-server", "stems-split"):
            assert output(directory / tool, "--version") == version, tool
        props = json.loads(output(directory / "stems-server", "--props", "--models-dir", models))
        assert props["service"] == "stems" and props["version"] == version, props
        assert any(d["backend"].upper().startswith("CPU") for d in props["devices"]), props
        if args.require_backend:
            assert any(d["backend"].lower() == args.require_backend
                       and d["type"] in {"gpu", "igpu", "integrated_gpu"}
                       for d in props["devices"]), props
        # A foreign program in a folder with no ggml, as a DAW is.
        abi_test = Path(scratch) / "stems-abi-test"
        shutil.copy2(args.abi_test, abi_test)
        print(output(abi_test, directory / "libstems.so", version))
        assert not list(models.iterdir()), "offline probes wrote into the models folder"
    info = json.loads((directory / "BUILD-INFO.json").read_text())
    assert info["service"] == "stems" and info["version"] == args.version, info
    assert info["platform"] == args.expect_platform and info["commit"] and info["ggml_commit"], info
    assert (directory / "libstems_v1.h").is_file()
    print("PASS extracted stems tools, libstems.so loaded from outside, devices and metadata")


if __name__ == "__main__":
    main()

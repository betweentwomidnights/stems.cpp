#!/usr/bin/env python3
"""Build flat Linux core/backend ZIPs of stems.cpp; Python is a build dependency only.

The Linux counterpart of ci/package-windows.ps1, adapted from sa3.cpp's and
acestep.cpp's ci/package-linux.py. The core carries libstems.so, stems-server,
stems-split and ggml's core and CPU variants as runtime-loaded libraries; a
backend ZIP adds one GPU backend beside them. Every ggml library is named
libstems-ggml*.so (GGML_LIBRARY_PREFIX), so a DAW that already loaded another
plugin's libggml cannot hand stems that copy, and every binary finds its siblings
through RPATH $ORIGIN. CUDA user libraries and the Vulkan loader/driver remain
system dependencies.

Example Spark smoke build:
  python3 ci/package-linux.py --version v0.1.7 --backends vulkan --require-gpu
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ("stems-server", "stems-split")
GPU_BACKENDS = ("cuda", "vulkan")


def run(*command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def output(*command):
    return subprocess.check_output(list(map(str, command)), text=True).strip()


def owned_directory(value, kind):
    path = (ROOT / value).resolve()
    if ROOT not in path.parents or path.parts[len(ROOT.parts)] in {
        ".git", "ggml", "src", "tools", "vendor", "docs", "tests", "ci"
    }:
        raise ValueError(f"{kind} must be a dedicated directory inside {ROOT}")
    marker = path / ".stems-linux-package"
    if path.exists() and not marker.is_file() and any(path.iterdir()):
        raise ValueError(f"refusing nonempty, unowned {kind}: {path}")
    path.mkdir(parents=True, exist_ok=True)
    marker.write_text("Owned by ci/package-linux.py\n", encoding="utf-8")
    return path


def cache_values(build):
    result = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        match = re.match(r"([^:#]+):[^=]+=(.*)", line)
        if match:
            result[match[1]] = match[2]
    return result


def is_elf(path):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


def elf_info(directory):
    """Record actual loader requirements and reject build-directory RPATHs."""
    needed = set()
    versions = {"GLIBC": set(), "GLIBCXX": set()}
    for path in directory.iterdir():
        if not is_elf(path):
            continue
        dynamic = output("readelf", "-d", path)
        for tag, value in re.findall(r"\((NEEDED|RPATH|RUNPATH)\).*?\[(.*?)\]", dynamic):
            if tag == "NEEDED":
                needed.add(value)
            elif value != "$ORIGIN":
                raise ValueError(f"{path.name} has non-relocatable {tag}: {value}")
        symbols = output("readelf", "--version-info", path)
        for family in versions:
            versions[family].update(re.findall(rf"\b{family}_(\d+(?:\.\d+)+)\b", symbols))
    return {
        "external_sonames": sorted(name for name in needed if not (directory / name).exists()),
        "symbol_versions": {family: max(items, key=lambda v: tuple(map(int, v.split('.'))))
                            for family, items in versions.items() if items},
    }


def write_zip(directory, destination):
    # Dereference the library aliases: hosts extract ordinary files, not symlinks.
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory).as_posix())


def unpack(archive_path, destination):
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents:
                raise ValueError(f"unsafe archive member: {member.filename}")
            archive.extract(member, destination)
            if not member.is_dir():
                target.chmod((member.external_attr >> 16) & 0o777)


def check_loader_paths(directory):
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    for path in directory.iterdir():
        if not is_elf(path):
            continue
        loaded = subprocess.check_output(["ldd", str(path)], env=env, text=True)
        if "not found" in loaded:
            raise ValueError(f"unresolved package dependencies for {path}:\n{loaded}")
        if re.search(r"\blibggml[^ ]*\.so", loaded):
            raise ValueError(f"{path.name} links an unprefixed libggml:\n{loaded}")
        for name, resolved in re.findall(r"(libstems[^ ]*\.so[^ ]*) => (\S+)", loaded):
            if Path(resolved).resolve().parent != directory.resolve():
                raise ValueError(f"{path.name} loads {name} outside the extracted package: {resolved}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True)
    parser.add_argument("--backends", nargs="*", choices=GPU_BACKENDS, default=["vulkan"])
    parser.add_argument("--cuda-arch", default="")
    parser.add_argument("--build-dir", default="build-dist-linux")
    parser.add_argument("--out-dir", default="dist-linux")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--require-gpu", action="store_true", help="require a real device in each selected backend's --props")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in {"aarch64", "x86_64"}:
        parser.error("build natively on ARM64 or x64 Linux")
    if not re.fullmatch(r"v\d+\.\d+\.\d+", args.version) or args.jobs < 1:
        parser.error("use --version vX.Y.Z and a positive --jobs")
    if args.cuda_arch and "cuda" not in args.backends:
        parser.error("--cuda-arch requires the CUDA backend")
    if args.require_gpu and not args.backends:
        parser.error("--require-gpu needs at least one backend")
    for command in ("cmake", "ctest", "git", "readelf", "ldd"):
        if not shutil.which(command):
            parser.error(f"missing build dependency: {command}")
    os.chdir(ROOT)
    declared = re.search(r"project\(stems_cpp VERSION (\d+\.\d+\.\d+)", (ROOT / "CMakeLists.txt").read_text())
    if not declared or args.version != "v" + declared[1]:
        parser.error("--version must match CMakeLists.txt project(VERSION)")
    dirty = bool(output("git", "status", "--porcelain"))
    if args.require_clean and dirty:
        parser.error("--require-clean: commit or remove source changes first")
    pin = output("git", "rev-parse", "HEAD:ggml")
    ggml = output("git", "-C", "ggml", "rev-parse", "HEAD")
    if ggml != pin or output("git", "-C", "ggml", "status", "--porcelain"):
        parser.error("ggml must be clean at the pinned submodule revision")
    build_candidate, out_candidate = (ROOT / args.build_dir).resolve(), (ROOT / args.out_dir).resolve()
    if (build_candidate == out_candidate or build_candidate in out_candidate.parents
            or out_candidate in build_candidate.parents):
        parser.error("build and output directories must be separate")
    build = owned_directory(args.build_dir, "build directory")
    destination = owned_directory(args.out_dir, "output directory")
    old_cache = cache_values(build) if (build / "CMakeCache.txt").is_file() else {}
    compilers = {
        "C": os.environ.get("CC") or old_cache.get("CMAKE_C_COMPILER") or shutil.which("cc"),
        "CXX": os.environ.get("CXX") or old_cache.get("CMAKE_CXX_COMPILER") or shutil.which("c++"),
    }
    if not all(compilers.values()):
        parser.error("C and C++ compilers are required; select them with CC/CXX")
    # ggml's full ARM variant set includes SME, which Ubuntu 24.04's default GCC 13 cannot build.
    if platform.machine() == "aarch64":
        for language, compiler in compilers.items():
            probe = subprocess.run([compiler, "-march=armv9.2-a+sme", "-x",
                                    "c" if language == "C" else "c++", "-fsyntax-only", "-"],
                                   input="int main(void) { return 0; }", text=True, capture_output=True)
            if probe.returncode:
                parser.error(f"{compiler} cannot build ggml's ARM SME variants. "
                             "Select capable compilers with CC/CXX (tested with GCC 14).\n" + probe.stderr)
    platform_key = "linux-arm64" if platform.machine() == "aarch64" else "linux-x64"
    configure = ["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                 "-DGGML_NATIVE=OFF",
                 "-DGGML_BACKEND_DL=ON",
                 "-DGGML_CPU_ALL_VARIANTS=ON",
                 "-DGGML_METAL=OFF",
                 "-DSTEMS_BUILD_TOOLS=ON",
                 "-DBUILD_TESTING=ON",
                 "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                 "-DCMAKE_INSTALL_RPATH=$ORIGIN",
                 "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF",
                 f"-DPython3_EXECUTABLE={sys.executable}"]
    for backend in GPU_BACKENDS:
        configure.append(f"-DSTEMS_{backend.upper()}={'ON' if backend in args.backends else 'OFF'}")
    for language, compiler in compilers.items():
        configure.append(f"-DCMAKE_{language}_COMPILER={compiler}")
    configure += ([f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}"] if args.cuda_arch
                  else ["-U", "CMAKE_CUDA_ARCHITECTURES"])
    run(*configure)
    if any(old_cache.get(f"CMAKE_{language}_COMPILER") not in {None, compiler}
           for language, compiler in compilers.items()):
        # Changing compilers makes CMake clear cached policy values; reapply them.
        run(*configure)
    run("cmake", "--build", build, "--parallel", args.jobs)
    if not args.skip_tests:
        run("ctest", "--test-dir", build, "--output-on-failure", "--timeout", "120")
    binaries = build / "bin"
    for tool in TOOLS:
        reported = output(binaries / tool, "--version")
        if "v" + reported != args.version:
            raise ValueError(f"{tool} reports {reported} but the package is {args.version}")

    package = build / "package"
    if package.exists():
        shutil.rmtree(package)
    core = package / "core"
    core.mkdir(parents=True)
    for tool in TOOLS:
        shutil.copy2(binaries / tool, core / tool)
    for library in [*binaries.glob("libstems.so*"), *binaries.glob("libstems-ggml*.so*")]:
        if not any(library.name.startswith(f"libstems-ggml-{backend}.") for backend in GPU_BACKENDS):
            shutil.copy2(library, core / library.name)
    for required in ("libstems.so", "libstems-ggml.so", "libstems-ggml-base.so"):
        if not list(core.glob(required + "*")):
            raise ValueError(f"missing {required} from the core package")
    if not list(core.glob("libstems-ggml-cpu*.so*")):
        raise ValueError("missing CPU backend libraries from the core package")
    if list(core.glob("libggml*")):
        raise ValueError("an unprefixed libggml reached the core package")
    for source, name in [("LICENSE", "LICENSE"),
                         ("src/libstems_v1.h", "libstems_v1.h"),
                         ("docs/RUNTIME_RELEASE.md", "RUNTIME_README.md"),
                         ("docs/THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
                         ("ggml/LICENSE", "LICENSE-ggml.txt"),
                         ("vendor/cpp-httplib/LICENSE", "LICENSE-cpp-httplib.txt"),
                         ("vendor/yyjson/LICENSE", "LICENSE-yyjson.txt")]:
        shutil.copy2(ROOT / source, core / name)
    stages = {"core": core}
    for backend in dict.fromkeys(args.backends):
        stage = package / backend
        stage.mkdir()
        libraries = list(binaries.glob(f"libstems-ggml-{backend}.so*"))
        if not libraries:
            raise ValueError(f"the {backend} backend library was not built")
        for library in libraries:
            shutil.copy2(library, stage / library.name)
        stages[backend] = stage
    cache = cache_values(build)
    info = {
        "service": "stems", "version": args.version, "platform": platform_key,
        "commit": output("git", "rev-parse", "HEAD"), "ggml_commit": ggml, "dirty": dirty,
        "build_flavor": "gpu-smoke" if args.cuda_arch else "portable",
        "backends": list(dict.fromkeys(args.backends)),
        "cuda_architecture_policy": (args.cuda_arch or "ggml-default") if "cuda" in args.backends else None,
        "cuda_compiler": output(cache["CMAKE_CUDA_COMPILER"], "--version") if "cuda" in args.backends else None,
        "compiler": output(cache["CMAKE_CXX_COMPILER"], "--version").splitlines()[0],
        "cmake": output("cmake", "--version").splitlines()[0],
        "system": platform.freedesktop_os_release(),
        "dependencies": {name: elf_info(stage) for name, stage in stages.items()},
        "runtime_policy": "system GPU driver/loader and CUDA user libraries; no redistributables bundled",
        "tests_run": not args.skip_tests, "gpu_probe_required": args.require_gpu,
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (core / "BUILD-INFO.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    archives = {}
    for name, stage in stages.items():
        archive = package / f"stems-{args.version}-{platform_key}-{name}.zip"
        write_zip(stage, archive)
        archives[name] = archive
    # Test the archives after extracting them elsewhere, not the build or the stage:
    # core alone, then core with each backend, the way a host installs them.
    for backend in [None, *dict.fromkeys(args.backends)]:
        with tempfile.TemporaryDirectory(prefix=f"check-{backend or 'core'}-", dir=build) as scratch:
            extracted = Path(scratch) / "runtime"
            extracted.mkdir()
            unpack(archives["core"], extracted)
            if backend:
                unpack(archives[backend], extracted)
            check_loader_paths(extracted)
            command = [sys.executable, ROOT / "ci/check-linux-package.py", extracted, args.version,
                       "--expect-platform", platform_key, "--abi-test", binaries / "stems-abi-test"]
            if backend and args.require_gpu:
                command += ["--require-backend", backend]
            run(*command)
    # Only completed, verified archives reach dist; the checksum file is written last.
    sums_text = "".join(f"{sha256(path)}  {path.name}\n" for path in archives.values())
    for archive in archives.values():
        shutil.copy2(archive, destination / (archive.name + ".partial"))
        (destination / (archive.name + ".partial")).replace(destination / archive.name)
    sums = destination / f"SHA256SUMS-{platform_key}"
    sums.with_suffix(".partial").write_text(sums_text, encoding="utf-8")
    sums.with_suffix(".partial").replace(sums)
    print(f"Verified packages: {destination}")
    run("du", "-sh", build, destination)


if __name__ == "__main__":
    main()

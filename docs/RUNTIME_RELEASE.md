# stems.cpp native runtime release

The Windows release follows the yuey.cpp and sa3.cpp package contract
(gary-localhost-installer `docs/native-runtime-packages.md`). A host installs
the core archive and one backend archive and verifies their SHA-256 hashes.
Direct users take one standalone archive with both backends and the CUDA
runtime included. Model weights are downloaded separately from Hugging Face
(`models.cmd`, or the URLs in [DISTRIBUTION.md](DISTRIBUTION.md)).

| Archive | Contents |
|---|---|
| `stems-<tag>-windows-x64-core.zip` | `stems.dll` and `libstems_v1.h`, `stems-server.exe`, `stems-split.exe`, ggml core and CPU variants, licenses, `BUILD-INFO.json`, and this document |
| `stems-<tag>-windows-x64-cuda.zip` | `stems-ggml-cuda.dll` for NVIDIA GPUs (needs the CUDA 12 runtime beside it or on `PATH`) |
| `stems-<tag>-windows-x64-vulkan.zip` | `stems-ggml-vulkan.dll` for AMD, Intel, or Vulkan-capable NVIDIA GPUs |
| `stems-<tag>-windows-x64-standalone.zip` | Core, both GPU backends, CUDA runtime and EULA, `models.cmd`, and a startup guide; for direct use |

The archives are flat at their root: unpack core, then one backend over it,
into the same folder. `SHA256SUMS` covers the four release archives. The
release is the manifest: a host pins a tag and checks each zip against that
tag's `SHA256SUMS`.

All binaries link the Visual C++ 2015–2022 runtime dynamically, OpenMP
(`VCOMP140.DLL`) included, as sa3.cpp's and yuey.cpp's do. A host installer
should make sure the VC++ redistributable is present.

## Embedding: gary4juce

gary4juce downloads core plus the Vulkan backend (Vulkan covers NVIDIA, AMD and
Intel without the ~700 MB CUDA runtime) into a per-user folder and loads the
library from there, so stems.cpp never ships inside the plugin bundle:

```c
HMODULE m = LoadLibraryExW(L"C:\\...\\stems\\v0.1.0\\stems.dll", NULL,
                           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
const stems_api_v1* api = ((const stems_api_v1* (*)(uint32_t))
                           GetProcAddress(m, "stems_get_api"))(STEMS_ABI_VERSION_1);
```

`LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` resolves `stems-ggml.dll` and
`stems-ggml-base.dll` from the runtime folder. `stems.dll` then registers the ggml backends (the CPU
variants and `stems-ggml-vulkan.dll`) from its own folder, not from the DAW's
executable folder or working directory. `libstems_v1.h` documents the rest:
`context_config.device` is NULL for the best GPU (most memory) with CPU
fallback, or `"cpu"`;
a context is not reentrant; `separate()` blocks and calls the progress and
cancel callbacks between segments, so run it on a worker thread.
`runtime_version()` returns `stems.cpp <version> (libstems abi 1)`.

Every ggml library is named `stems-ggml*.dll` (`GGML_LIBRARY_PREFIX`). Windows
binds an import to any already-loaded DLL of the same name, from whatever
folder, so with plain `ggml.dll` names a DAW that had already loaded another
plugin's ggml (FoundationKeys 0.1.x ships `ggml.dll` and `ggml-base.dll`)
would hand stems that older copy: measured, the stems then ran on
FoundationKeys' `ggml-base.dll` and moved to 71 dB from a clean run. With the
prefix the same session is bit-identical to running stems alone.

## macOS

`stems-<tag>-macos-universal.zip` is a single archive for Apple Silicon and Intel
Macs: `libstems.dylib` and `libstems_v1.h`, `stems-server`, `stems-split`, `models.sh`,
licenses, and `BUILD-INFO.json`. `SHA256SUMS-macos` covers it, kept apart from the
Windows `SHA256SUMS` so the two jobs never write one file.

There are no backend archives and no ggml libraries. ggml is linked statically, so
`libstems.dylib` holds ggml, the CPU backend (Accelerate) and the Metal backend with
its shader source embedded, and exports only `stems_get_api`. Nothing in it can bind to
another plugin's ggml. It links only system frameworks and runs on macOS 13.3 or later.

Every binary is universal. The arm64 slice runs on the Metal GPU. The x86_64 slice is
CPU only (ggml's Metal backend needs an Apple GPU) and needs AVX2, which every Intel Mac
that runs macOS 13.3 has; expect HTDemucs at around half realtime there, and the
RoFormers much slower. dyld picks the slice that matches the host process.

```c
void* lib = dlopen("/Users/.../stems/v0.1.0/libstems.dylib", RTLD_NOW | RTLD_LOCAL);
const stems_api_v1* api = ((const stems_api_v1* (*)(uint32_t))
                           dlsym(lib, "stems_get_api"))(STEMS_ABI_VERSION_1);
```

`context_config.device` NULL picks the Metal GPU on Apple Silicon and the CPU on Intel. The binaries are signed with a
Developer ID and the hardened runtime, and the zip is notarized. A zip of bare
binaries cannot be stapled, so Gatekeeper checks the notarization online.

## Service: gary4local

`stems-server.exe` answers `--version` (the bare version) and `--props` (the
ggml devices, and which file each model name resolves to in the models
directory) without binding a port or loading a model; `GET /props` returns
the same JSON while it runs. The package script rejects a tag that disagrees
with the compiled version. For supervised launches, `STEMS_PORT` (default
8010), `STEMS_HOST`, `STEMS_MODELS_DIR`, and `STEMS_DEVICE` set the port, bind
address, model directory, and device (`cpu` forces the CPU; otherwise the best
GPU, which `STEMS_GPU` can narrow by index or name). The server defaults to loopback;
callers that expose it to a network must supply their own access controls.

## Build and publish

On an Apple Silicon Mac with Xcode and CMake (macOS 15, so Rosetta can run the x86_64
slice's checks):

```bash
ci/package-macos.sh --version v0.1.0 --check-model htdemucs-42M-v1.0-F16.gguf
```

It builds the arm64 (Metal) and x86_64 (CPU) slices in separate trees, runs CTest on
both, joins them with lipo, and checks that each slice exports only `stems_get_api`,
links only the system, and that Metal is in the arm64 slice alone. It signs when
`STEMS_SIGN_IDENTITY` is set, separates a second of audio through the staged dylib
from a folder outside the package, once from an arm64 process and once from an x86_64
one under Rosetta, then zips and notarizes when the `STEMS_NOTARY_*` variables are set. In CI
the Developer ID certificate and an App Store Connect API key come from the
`MACOS_CERT_P12`, `MACOS_CERT_PASSWORD`, `APPLE_TEAM_ID`, `APPLE_NOTARY_KEY_P8`,
`APPLE_NOTARY_KEY_ID` and `APPLE_NOTARY_ISSUER_ID` secrets; a tagged build without
them fails. GitHub's macOS runners have a paravirtualized GPU that ggml's Metal backend
does not run reliably on, so there the arm64 separation check runs on the CPU. Metal itself
is validated on real hardware (docs/PARITY.md).

On Windows with Visual Studio 2022, CUDA Toolkit 12.8, and Vulkan SDK:

```powershell
.\ci\package-windows.ps1 -Version v0.1.0
# Add a separate cudart zip for testing gary4local's shared-runtime install:
.\ci\package-windows.ps1 -Version v0.1.0 -CudaRuntime
```

For a quick local check of the core package without GPU toolchains, use
`-CpuOnly`. For a local GPU packaging check without compiling every CUDA
architecture, use `-CudaArch native -BuildDir build-dist-native -OutDir dist-native`.
Both outputs are for smoke testing and are not published releases. Keep the
checkout path short: the Vulkan shader build exceeds `MAX_PATH` under a deep
folder.

The script configures `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`, and
`GGML_NATIVE=OFF`, runs CTest, checks both executables' versions, and stages
the archives. Before zipping it checks the staged core from a scratch folder
outside it: `stems-server --version` and `--props` must run with only what core
ships, and `stems-abi-test` must load `stems.dll` by path and find its CPU
backend, the way a plugin does. It then writes SHA-256 sums. `BUILD-INFO.json`
records the service, version, source and ggml commits, toolchains, and whether
local tracked files differed from the commit.

CTest runs without any model: the DSP round trips, the C ABI as a host sees it,
the `--version`/`--props` contract, and model-file resolution.

## Release checks

1. **Dry run:** dispatch `release.yml` with no tag, on the intended branch:
   `gh workflow run release.yml -R betweentwomidnights/stems.cpp --ref main`.
   `-f platforms=macos` (or `windows`) builds one platform only; adding macOS to an
   existing release is `-f tag=v0.1.0 -f platforms=macos`.
   CI builds the full portable package and retains the four zips plus
   `SHA256SUMS` as a workflow artifact for 14 days. It creates no release.
2. **Install test:** download that artifact and check `SHA256SUMS`. Unpack
   core plus Vulkan into a fresh folder, run `stems-server --props` there, and
   separate a song through `stems.dll` from a process outside the folder
   (`stems-abi-test <folder>\stems.dll <version> <model>` from a build tree).
   Test the standalone zip directly with CUDA and Vulkan.
3. **Publish:** tag and publish the verified commit as `v0.1.0`. The release
   event runs the same package script on that tag, attests each zip, and
   attaches the four zips and `SHA256SUMS`. A dispatch with an existing tag
   can replace missing assets after a runner failure, before a consumer pins it.
4. **Pin and test:** verify the published checksums and attestation, then pin
   the tag and its hashes in gary4juce. Test installation from those published
   URLs.

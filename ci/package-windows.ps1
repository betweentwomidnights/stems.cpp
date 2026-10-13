# ABOUTME: Builds split Windows packages for gary4juce/gary4local and one standalone zip
# ABOUTME: with both GPU backends and the CUDA runtime, plus SHA256SUMS.
#
# Same package contract as sa3.cpp and yuey.cpp (gary-localhost-installer
# docs/native-runtime-packages.md). The backends are built as dynamic libraries
# (GGML_BACKEND_DL) and the CPU backend in every instruction-set variant
# (GGML_CPU_ALL_VARIANTS), so one core package runs on any x64 machine and a GPU
# backend is a single DLL unpacked beside it: unpack core, unpack one backend
# over it, run.
#
# The core is also what a plugin downloads: gary4juce loads stems.dll from the
# unpacked folder (LoadLibraryEx + stems_get_api), and stems.dll registers the
# ggml backends from its own folder, not the DAW's. The script checks exactly
# that before zipping, from a folder outside the package.
#
# GGML_NATIVE is off so nothing is tuned to the machine that built it, and no
# CUDA architecture list is passed: with native off, ggml's own default covers
# Maxwell through Blackwell as PTX plus real code for the common cards.
# Gary4local installs the shared CUDA runtime once from its own release. The
# standalone zip includes it so direct users need only one archive. -CudaRuntime
# also emits a separate runtime zip for local supervisor install testing.
#
# The same script runs in .github/workflows/release.yml and on a developer
# machine, so a package built by hand is the package CI would have built.
#
# Usage:
#   ci\package-windows.ps1 -Version v0.1.0 [-BuildDir build-dist] [-OutDir dist] [-SkipTests] [-CudaRuntime]
#   ci\package-windows.ps1 -Version v0.1.0 -CpuOnly   # local packaging smoke check
#   ci\package-windows.ps1 -Version v0.1.0 -CudaArch native -BuildDir build-dist-native -OutDir dist-native
#
# -CheckModel PATH.gguf also separates a second of audio through the staged
# stems.dll on the CPU; without it the outside-the-folder check stops short of
# loading a backend.
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$BuildDir = "build-dist",
    [string]$OutDir = "dist",
    [switch]$SkipTests,
    [switch]$CpuOnly,
    [switch]$CudaRuntime,
    [string]$CudaArch = "",
    [string]$CheckModel = "",
    [int]$Jobs = 4
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# The script clears its staging directory before each run. Keep that deletion
# strictly inside this checkout even if a caller passes an absolute BuildDir.
$rootPrefix = [System.IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
$buildInput = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $root $BuildDir }
$outInput = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $root $OutDir }
$buildPath = [System.IO.Path]::GetFullPath($buildInput)
$outPath = [System.IO.Path]::GetFullPath($outInput)
if (-not $buildPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "BuildDir must be inside $root"
}
if (-not $outPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutDir must be inside $root"
}
if ($Version -notmatch '^v\d+\.\d+\.\d+$') { throw "Version must look like v0.1.0" }

function Fail([string]$message) {
    Write-Error "package-windows: $message"
    exit 1
}

function Invoke-Checked([string]$program, [string[]]$arguments) {
    & $program @arguments
    if ($LASTEXITCODE -ne 0) {
        Fail "$program exited with $LASTEXITCODE"
    }
}

# stdout of a native program, judged by its exit code alone. ggml logs to
# stderr, which Windows PowerShell 5.1 would otherwise turn into a terminating
# error under ErrorActionPreference=Stop once output is redirected.
function Get-Output([string]$program, [string[]]$arguments) {
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = & $program @arguments 2>$null
    } finally {
        $ErrorActionPreference = $saved
    }
    if ($LASTEXITCODE -ne 0) { Fail "$program $($arguments -join ' ') exited with $LASTEXITCODE" }
    return ($out -join "`n").Trim()
}

# The Visual C++ runtime, carried app-local. Every binary links it
# dynamically, OpenMP (vcomp140) included. Windows looks in an executable's
# own folder before System32, so copies beside it win over whatever the
# machine has, and nobody needs an administrator install of the
# redistributable. They come from the redistributable folder of the Visual
# Studio that CMake built the package with, and must be at least as new as its
# compiler: binaries from a newer toolset can crash on an older msvcp140
# (std::mutex changed in 14.40).
$vcRuntimeNames = "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcomp140.dll"

function Get-VcRuntime([string]$buildDir) {
    # Everything comes from the build's own configuration, not from whichever
    # Visual Studio or CMake happens to be newest: the cache names the CMake
    # that configured it and the Visual Studio instance it used, and that
    # CMake's own version folder holds the compiler it found.
    $cache = Get-Content (Join-Path $buildDir "CMakeCache.txt")
    function Cached([string]$name) {
        $line = $cache | Where-Object { $_ -match "^$([regex]::Escape($name)):[A-Z]+=(.+)$" } | Select-Object -First 1
        if (-not $line) { Fail "$buildDir\CMakeCache.txt has no $name" }
        return ($line -replace "^[^=]+=", "")
    }
    $instance = [System.IO.Path]::GetFullPath((Cached "CMAKE_GENERATOR_INSTANCE"))
    $cmakeVersion = (& (Cached "CMAKE_COMMAND") -E capabilities | ConvertFrom-Json).version.string
    $compilerFile = Join-Path $buildDir "CMakeFiles\$cmakeVersion\CMakeCXXCompiler.cmake"
    if (-not (Test-Path $compilerFile)) { Fail "$compilerFile is missing; configure $buildDir with CMake $cmakeVersion" }
    $record = Get-Content $compilerFile -Raw
    if ($record -notmatch 'set\(CMAKE_CXX_COMPILER "([^"]+)"\)') { Fail "$compilerFile names no C++ compiler" }
    $compiler = [System.IO.Path]::GetFullPath($Matches[1])
    if (-not $compiler.StartsWith($instance.TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        Fail "the compiler $compiler is not part of the Visual Studio at $instance"
    }
    # The compiler reports 19.MM; its runtime is 14.MM.
    if ($record -notmatch 'set\(CMAKE_CXX_COMPILER_VERSION "19\.(\d+)') { Fail "$compilerFile has no MSVC version" }
    $toolsetMinor = [int]$Matches[1]

    $versionFile = Join-Path $instance "VC\Auxiliary\Build\Microsoft.VCRedistVersion.default.txt"
    if (-not (Test-Path $versionFile)) { Fail "$instance has no VC redistributable version file" }
    $redist = Join-Path $instance "VC\Redist\MSVC\$((Get-Content $versionFile -Raw).Trim())\x64"

    $files = @()
    foreach ($name in $vcRuntimeNames) {
        $folder = if ($name -eq "vcomp140.dll") { "Microsoft.VC143.OpenMP" } else { "Microsoft.VC143.CRT" }
        $file = Join-Path $redist "$folder\$name"
        if (-not (Test-Path $file)) { Fail "$file is missing; install the MSVC v143 redistributable files" }
        $info = (Get-Item $file).VersionInfo
        if ($info.FileMajorPart -ne 14 -or $info.FileMinorPart -lt $toolsetMinor) {
            Fail "$name $($info.FileVersion) is older than the 14.$toolsetMinor compiler that built the package"
        }
        $files += $file
    }
    return $files
}

# Records every DLL a process loads, by the path Windows actually opened, by
# running it to exit as its debugger. Unlike listing a live process's modules
# it can't miss a short-lived one, and it sees backends loaded at run time.
if (-not ("DllLoadRecorder" -as [type])) { Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

public static class DllLoadRecorder
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    struct STARTUPINFO
    {
        public int cb;
        public string lpReserved, lpDesktop, lpTitle;
        public int dwX, dwY, dwXSize, dwYSize, dwXCountChars, dwYCountChars, dwFillAttribute, dwFlags;
        public short wShowWindow, cbReserved2;
        public IntPtr lpReserved2, hStdInput, hStdOutput, hStdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct PROCESS_INFORMATION { public IntPtr hProcess, hThread; public int dwProcessId, dwThreadId; }

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern bool CreateProcessW(string application, StringBuilder commandLine, IntPtr processAttributes,
        IntPtr threadAttributes, bool inheritHandles, uint flags, IntPtr environment, string directory,
        ref STARTUPINFO startup, out PROCESS_INFORMATION process);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool WaitForDebugEvent(byte[] debugEvent, uint milliseconds);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool ContinueDebugEvent(int processId, int threadId, uint status);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool TerminateProcess(IntPtr process, uint exitCode);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern uint GetFinalPathNameByHandleW(IntPtr file, StringBuilder path, uint size, uint flags);

    const uint DEBUG_ONLY_THIS_PROCESS = 0x2, CREATE_NO_WINDOW = 0x08000000;
    const uint DBG_CONTINUE = 0x00010002, DBG_EXCEPTION_NOT_HANDLED = 0x80010001;
    const uint EXCEPTION_DEBUG_EVENT = 1, CREATE_PROCESS_DEBUG_EVENT = 3, EXIT_PROCESS_DEBUG_EVENT = 5,
        LOAD_DLL_DEBUG_EVENT = 6, EXCEPTION_BREAKPOINT = 0x80000003;
    // DEBUG_EVENT on x64: code, process id and thread id, then the union at 16.
    const int UNION = 16;

    public static List<string> Run(string exe, string commandLine, string directory, int timeoutMs, out int exitCode)
    {
        var startup = new STARTUPINFO();
        startup.cb = Marshal.SizeOf(startup);
        PROCESS_INFORMATION process;
        if (!CreateProcessW(exe, new StringBuilder(commandLine), IntPtr.Zero, IntPtr.Zero, false,
                DEBUG_ONLY_THIS_PROCESS | CREATE_NO_WINDOW, IntPtr.Zero, directory, ref startup, out process))
            throw new Win32Exception();
        var loaded = new List<string>();
        var debugEvent = new byte[256];
        int deadline = Environment.TickCount + timeoutMs;
        try
        {
            while (true)
            {
                int left = deadline - Environment.TickCount;
                if (left <= 0)
                {
                    TerminateProcess(process.hProcess, 1);
                    throw new TimeoutException(exe + " did not exit within " + (timeoutMs / 1000) + " s");
                }
                if (!WaitForDebugEvent(debugEvent, (uint)left)) continue;
                uint code = BitConverter.ToUInt32(debugEvent, 0);
                int processId = BitConverter.ToInt32(debugEvent, 4), threadId = BitConverter.ToInt32(debugEvent, 8);
                uint status = DBG_CONTINUE;
                if (code == EXCEPTION_DEBUG_EVENT)
                {
                    // The loader's initial breakpoint is ours; anything else is the program's.
                    if (BitConverter.ToUInt32(debugEvent, UNION) != EXCEPTION_BREAKPOINT) status = DBG_EXCEPTION_NOT_HANDLED;
                }
                else if (code == CREATE_PROCESS_DEBUG_EVENT || code == LOAD_DLL_DEBUG_EVENT)
                {
                    // Both start with the image's file handle, which the debugger must close.
                    var file = new IntPtr(BitConverter.ToInt64(debugEvent, UNION));
                    if (file != IntPtr.Zero)
                    {
                        if (code == LOAD_DLL_DEBUG_EVENT)
                        {
                            var path = new StringBuilder(1024);
                            if (GetFinalPathNameByHandleW(file, path, (uint)path.Capacity, 0) > 0)
                                loaded.Add(path.ToString().Replace(@"\\?\", ""));
                        }
                        CloseHandle(file);
                    }
                }
                else if (code == EXIT_PROCESS_DEBUG_EVENT)
                {
                    exitCode = BitConverter.ToInt32(debugEvent, UNION);
                    ContinueDebugEvent(processId, threadId, DBG_CONTINUE);
                    return loaded;
                }
                ContinueDebugEvent(processId, threadId, status);
            }
        }
        finally
        {
            CloseHandle(process.hThread);
            CloseHandle(process.hProcess);
        }
    }
}
'@ }

# Run a staged executable from an unrelated working directory and fail unless
# every Visual C++ runtime DLL it loads, which must be all four (the arguments
# have to load a CPU backend, which brings in vcomp140), came from its own
# folder. The build machine has the runtime in System32 too, so this is what
# proves the package's copies are the ones that load.
function Assert-AppLocalVcRuntime([string]$exe, [string[]]$arguments) {
    $folder = Split-Path -Parent ([System.IO.Path]::GetFullPath($exe))
    $commandLine = (@("`"$exe`"") + ($arguments | ForEach-Object { "`"$_`"" })) -join " "
    $exitCode = 0
    $loaded = [DllLoadRecorder]::Run($exe, $commandLine, $env:TEMP, 120000, [ref]$exitCode)
    if ($exitCode -ne 0) { Fail "$(Split-Path -Leaf $exe) $($arguments -join ' ') exited with $exitCode under the DLL check" }
    foreach ($name in $vcRuntimeNames) {
        $paths = @($loaded | Where-Object { (Split-Path -Leaf $_) -ieq $name })
        if ($paths.Count -eq 0) { Fail "$(Split-Path -Leaf $exe) $($arguments -join ' ') never loaded $name" }
        foreach ($path in $paths) {
            if ((Split-Path -Parent $path) -ne $folder) { Fail "$(Split-Path -Leaf $exe) loaded $name from $path, not from the package" }
        }
    }
    Write-Host "app-local VC runtime: $(Split-Path -Leaf $exe) loads all four from its own folder"
}

# --- toolchain --------------------------------------------------------------

$cmake = (Get-Command cmake.exe -ErrorAction SilentlyContinue)
if ($cmake) {
    $cmake = $cmake.Source
    $ctest = Join-Path (Split-Path -Parent $cmake) "ctest.exe"
} else {
    foreach ($edition in "Community", "Professional", "Enterprise", "BuildTools") {
        $candidate = "C:\Program Files\Microsoft Visual Studio\2022\$edition\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        if (Test-Path $candidate) {
            $cmake = $candidate
            $ctest = Join-Path (Split-Path -Parent $candidate) "ctest.exe"
            break
        }
    }
}
if (-not $cmake) { Fail "CMake was not found on PATH or in Visual Studio 2022" }
if (-not $CpuOnly -and -not $env:CUDA_PATH) { Fail "CUDA_PATH is not set; install the CUDA Toolkit" }
if (-not $CpuOnly -and -not $env:VULKAN_SDK) { Fail "VULKAN_SDK is not set; install the Vulkan SDK" }
if ($CpuOnly -and $CudaArch) { Fail "-CudaArch cannot be used with -CpuOnly" }
if ($CpuOnly -and $CudaRuntime) { Fail "-CudaRuntime requires a GPU build" }
if ($CudaArch -and $BuildDir -eq "build-dist") { Fail "-CudaArch needs a separate -BuildDir to keep the portable release cache clean" }
if ($Jobs -lt 1) { Fail "-Jobs must be positive" }
if ($CheckModel) {
    if (-not (Test-Path -LiteralPath $CheckModel -PathType Leaf)) { Fail "-CheckModel $CheckModel does not exist" }
    $CheckModel = (Resolve-Path -LiteralPath $CheckModel).Path
}

# version.json ships with a full toolkit install; a trimmed CI install may only
# have the versioned folder, which is named for the same release (v12.8).
if (-not $CpuOnly) {
    $versionJson = Join-Path $env:CUDA_PATH "version.json"
    if (Test-Path $versionJson) {
        $cudaVersion = (Get-Content $versionJson -Raw | ConvertFrom-Json).cuda.version
    } else {
        $cudaVersion = (Split-Path -Leaf $env:CUDA_PATH).TrimStart("v")
    }
    if ($cudaVersion -notmatch "^\d+\.\d+") { Fail "cannot tell the CUDA version from $env:CUDA_PATH" }
    $cudaMajorMinor = ($cudaVersion -split "\.")[0..1] -join "."
    $cudaMajor = ($cudaVersion -split "\.")[0]
}

Write-Host "stems.cpp  $(git rev-parse --short HEAD)"
Write-Host "ggml       $(git -C ggml rev-parse HEAD)"
Write-Host "version    $Version"
if (-not $CpuOnly) {
    Write-Host "cuda       $cudaVersion ($env:CUDA_PATH)"
    Write-Host "vulkan     $env:VULKAN_SDK"
}

# --- build ------------------------------------------------------------------

$gpuBuild = if ($CpuOnly) { "OFF" } else { "ON" }
$configure = @(
    "-S", ".", "-B", $buildPath,
    "-G", "Visual Studio 17 2022", "-A", "x64",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_NATIVE=OFF",
    "-DGGML_BACKEND_DL=ON",
    "-DGGML_CPU_ALL_VARIANTS=ON",
    "-DGGML_METAL=OFF",
    "-DSTEMS_CUDA=$gpuBuild",
    "-DSTEMS_VULKAN=$gpuBuild",
    "-DSTEMS_BUILD_TOOLS=ON",
    "-DBUILD_TESTING=ON"
)
if ($CudaArch) { $configure += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch" }
Invoke-Checked $cmake $configure
Invoke-Checked $cmake @("--build", $buildPath, "--config", "Release", "--parallel", "$Jobs")

# The model-free tests run on the CPU backend, which here is loaded
# dynamically exactly as it will be on a user's machine.
if (-not $SkipTests) {
    Invoke-Checked $ctest @("--test-dir", $buildPath, "-C", "Release", "--output-on-failure")
}

$bin = Join-Path $buildPath "bin\Release"
foreach ($tool in "stems-server.exe", "stems-split.exe") {
    $reported = Get-Output (Join-Path $bin $tool) @("--version")
    if ("v$reported" -ne $Version) {
        Fail "$tool reports $reported but the package is $Version; update project(VERSION) in CMakeLists.txt"
    }
}

# --- stage ------------------------------------------------------------------

$stage = Join-Path $buildPath "package"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $outPath | Out-Null

function Stage([string]$name, [string[]]$patterns, [string]$from) {
    $dir = Join-Path $stage $name
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($pattern in $patterns) {
        $found = @(Get-ChildItem -Path $from -Filter $pattern -File)
        if ($found.Count -eq 0) { Fail "$name package: nothing matches $pattern in $from" }
        foreach ($file in $found) { Copy-Item $file.FullName $dir }
    }
    return $dir
}

$coreDir = Stage "core" @(
    "stems-server.exe",
    "stems-split.exe",
    "stems.dll",
    "stems-ggml.dll",
    "stems-ggml-base.dll",
    "stems-ggml-cpu-*.dll"
) $bin
Copy-Item (Join-Path $root "LICENSE") $coreDir
Copy-Item (Join-Path $root "src\libstems_v1.h") $coreDir
Copy-Item (Join-Path $root "docs\RUNTIME_RELEASE.md") (Join-Path $coreDir "RUNTIME_README.md")
Copy-Item (Join-Path $root "docs\THIRD_PARTY_NOTICES.md") (Join-Path $coreDir "THIRD_PARTY_NOTICES.md")
Copy-Item (Join-Path $root "ggml\LICENSE") (Join-Path $coreDir "LICENSE-ggml.txt")
Copy-Item (Join-Path $root "vendor\cpp-httplib\LICENSE") (Join-Path $coreDir "LICENSE-cpp-httplib.txt")
Copy-Item (Join-Path $root "vendor\yyjson\LICENSE") (Join-Path $coreDir "LICENSE-yyjson.txt")
$vcRuntime = Get-VcRuntime $buildPath
foreach ($file in $vcRuntime) { Copy-Item $file $coreDir }

$buildInfo = [ordered]@{
    service     = "stems"
    version     = $Version
    commit      = (git rev-parse HEAD).Trim()
    dirty       = [bool](git status --porcelain --untracked-files=no)
    ggml_commit = (git -C ggml rev-parse HEAD).Trim()
    platform    = "windows-x64"
    backends    = @()
    cuda        = $(if ($CpuOnly) { $null } else { $cudaVersion })
    vulkan_sdk  = $(if ($CpuOnly) { $null } else { Split-Path -Leaf $env:VULKAN_SDK })
    msvc_runtime = (Get-Item $vcRuntime[0]).VersionInfo.FileVersion
    built_utc   = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
}
if (-not $CpuOnly) { $buildInfo.backends = @("cuda", "vulkan") }
[System.IO.File]::WriteAllText(
    (Join-Path $coreDir "BUILD-INFO.json"),
    ($buildInfo | ConvertTo-Json) + "`n",
    (New-Object System.Text.UTF8Encoding($false)))

# A GPU backend that landed in the core zip would load on every machine, and
# one missing from its own zip would never load anywhere. Check both ways.
foreach ($backend in "cuda", "vulkan") {
    if (Test-Path (Join-Path $coreDir "stems-ggml-$backend.dll")) { Fail "stems-ggml-$backend.dll leaked into the core package" }
}
if (-not $CpuOnly) {
    $cudaDir = Stage "cuda" @("stems-ggml-cuda.dll") $bin
    $vulkanDir = Stage "vulkan" @("stems-ggml-vulkan.dll") $bin

    $cudartDir = Stage "cudart" @(
        "cudart64_$cudaMajor.dll",
        "cublas64_$cudaMajor.dll",
        "cublasLt64_$cudaMajor.dll"
    ) (Join-Path $env:CUDA_PATH "bin")
    Copy-Item (Join-Path $env:CUDA_PATH "EULA.txt") (Join-Path $cudartDir "NVIDIA-CUDA-EULA.txt")

    $standaloneDir = Join-Path $stage "standalone"
    New-Item -ItemType Directory -Force $standaloneDir | Out-Null
    foreach ($part in $coreDir, $cudaDir, $vulkanDir, $cudartDir) {
        Get-ChildItem -Path $part -File | Copy-Item -Destination $standaloneDir
    }
    Copy-Item (Join-Path $root "models.cmd") $standaloneDir
    Copy-Item (Join-Path $root "ci\standalone-README.txt") (Join-Path $standaloneDir "README.txt")
}

# --- check the staged core from outside it -----------------------------------
#
# Runs from a scratch folder that holds no ggml, the way a DAW runs: the
# executables must start with only what core ships, and stems.dll, loaded by
# absolute path from a foreign executable, must register its CPU backend from
# its own folder. Catches a DLL the core forgot as well as a loader regression.

$scratch = Join-Path ([System.IO.Path]::GetTempPath()) ("stems-package-check-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force $scratch | Out-Null
try {
    Copy-Item (Join-Path $bin "stems-abi-test.exe") $scratch
    Push-Location $scratch
    try {
        $server = Join-Path $coreDir "stems-server.exe"
        $reported = Get-Output $server @("--version")
        if ("v$reported" -ne $Version) { Fail "staged stems-server.exe reports $reported" }
        $props = Get-Output $server @("--props", "--models-dir", $scratch) | ConvertFrom-Json
        if (-not ($props.devices | Where-Object { $_.backend -eq "CPU" })) {
            Fail "staged stems-server.exe --props lists no CPU device"
        }
        Assert-AppLocalVcRuntime $server @("--props", "--models-dir", $scratch)
        $abiArgs = @((Join-Path $coreDir "stems.dll"), $Version.TrimStart("v"))
        if ($CheckModel) { $abiArgs += $CheckModel }
        # The core alone has only CPU backends, so this separation runs on the CPU
        # backend stems.dll found beside itself.
        Write-Host (Get-Output (Join-Path $scratch "stems-abi-test.exe") $abiArgs)
    } finally {
        Pop-Location
    }
} finally {
    Remove-Item -Recurse -Force $scratch -ErrorAction SilentlyContinue
}
Write-Host "staged core checked from outside its folder"

# --- zip and checksum -------------------------------------------------------

Add-Type -AssemblyName System.IO.Compression.FileSystem

$archives = [ordered]@{
    "stems-$Version-windows-x64-core.zip" = $coreDir
}
if (-not $CpuOnly) {
    $archives["stems-$Version-windows-x64-cuda.zip"] = $cudaDir
    $archives["stems-$Version-windows-x64-vulkan.zip"] = $vulkanDir
    $archives["stems-$Version-windows-x64-standalone.zip"] = $standaloneDir
    if ($CudaRuntime) { $archives["cudart-$cudaMajorMinor-windows-x64.zip"] = $cudartDir }
}

$existing = @(Get-ChildItem -LiteralPath $outPath -File)
if (@($existing | Where-Object { $_.Name -notmatch '^(stems-v\d+\.\d+\.\d+-windows-x64-(core|cuda|vulkan|standalone)\.zip|cudart-\d+\.\d+-windows-x64\.zip|SHA256SUMS)$' }).Count) {
    Fail "OutDir contains files that are not stems.cpp package outputs: $outPath"
}
if ($existing.Count) { $existing | Remove-Item -Force }

$sums = New-Object System.Text.StringBuilder
foreach ($entry in $archives.GetEnumerator()) {
    $zip = Join-Path $outPath $entry.Key
    if (Test-Path $zip) { Remove-Item -Force $zip }
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        (Resolve-Path $entry.Value).Path, $zip,
        [System.IO.Compression.CompressionLevel]::Optimal, $false)
    $hash = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLowerInvariant()
    [void]$sums.Append("$hash  $($entry.Key)`n")
    $megabytes = [math]::Round((Get-Item $zip).Length / 1MB, 1)
    Write-Host ("{0,-46} {1,8} MB  {2}" -f $entry.Key, $megabytes, $hash)
}

# LF endings and no BOM, so `sha256sum -c SHA256SUMS` works as-is.
[System.IO.File]::WriteAllText(
    (Join-Path $outPath "SHA256SUMS"),
    $sums.ToString(),
    (New-Object System.Text.UTF8Encoding($false)))

Write-Host "packages -> $outPath"

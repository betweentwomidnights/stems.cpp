#!/usr/bin/env bash
# ABOUTME: Builds the macOS arm64 package: one self-contained libstems.dylib with Metal,
# ABOUTME: stems-server and stems-split, signed and notarized when credentials are given.
#
# The macOS counterpart of ci/package-windows.ps1, and the same package contract
# (gary-localhost-installer docs/native-runtime-packages.md), with one difference:
# there is one archive, not core plus backends. ggml is linked statically (the default
# build), so libstems.dylib carries ggml, the CPU backend and the Metal backend with its
# shader source embedded, and exports only stems_get_api. There are no ggml dylibs for
# another plugin's copy to collide with, and nothing to unpack beside it.
#
# gary4juce downloads the zip and dlopens libstems.dylib from the unpacked folder. The
# script checks exactly that before zipping, from a folder outside the package, then
# signs (Developer ID, hardened runtime) and notarizes when the environment provides:
#
#   STEMS_SIGN_IDENTITY   codesign identity (name or SHA-1) in an unlocked keychain
#   STEMS_NOTARY_KEY      path to an App Store Connect API key (.p8)
#   STEMS_NOTARY_KEY_ID   its key ID
#   STEMS_NOTARY_ISSUER   its issuer ID
#
# Without them the binaries keep the linker's ad-hoc signature: fine for a local check,
# not for publishing. --require-signing makes their absence an error (release builds).
# A zip of bare binaries cannot be stapled; Gatekeeper looks the notarization up online.
#
# Usage:
#   ci/package-macos.sh --version v0.1.0 [--build-dir build-dist-macos] [--out-dir dist]
#                       [--check-model PATH.gguf] [--skip-tests] [--require-signing] [--jobs N]
#
# --check-model also separates a second of audio through the staged libstems.dylib. It
# runs on the Metal GPU, except on a paravirtualized one (GitHub's macOS runners), where
# ggml's Metal backend is not reliable and the check runs on the CPU instead.
set -euo pipefail

VERSION=""
BUILD_DIR="build-dist-macos"
OUT_DIR="dist"
CHECK_MODEL=""
SKIP_TESTS=0
REQUIRE_SIGNING=0
JOBS=4
DEPLOYMENT_TARGET="13.3"

fail() { echo "package-macos: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift ;;
    --build-dir) BUILD_DIR="$2"; shift ;;
    --out-dir) OUT_DIR="$2"; shift ;;
    --check-model) CHECK_MODEL="$2"; shift ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --require-signing) REQUIRE_SIGNING=1 ;;
    --jobs) JOBS="$2"; shift ;;
    -h|--help) sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
  shift
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

[[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "--version must look like v0.1.0"
[ "$(uname -s)" = Darwin ] || fail "this script builds the macOS package; run it on a Mac"
[ "$(uname -m)" = arm64 ] || fail "build on Apple Silicon: the package is arm64"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail "--jobs must be positive"

# The script clears its staging and output folders, so keep both inside this checkout.
abspath() { case "$1" in /*) echo "$1" ;; *) echo "$ROOT/$1" ;; esac; }
BUILD_PATH="$(abspath "$BUILD_DIR")"
OUT_PATH="$(abspath "$OUT_DIR")"
for p in "$BUILD_PATH" "$OUT_PATH"; do
  case "$p/" in "$ROOT"/?*) ;; *) fail "$p must be inside $ROOT" ;; esac
  case "$p" in *..*) fail "$p must not contain .." ;; esac
done

if [ -n "$CHECK_MODEL" ]; then
  [ -f "$CHECK_MODEL" ] || fail "--check-model $CHECK_MODEL does not exist"
  CHECK_MODEL="$(cd "$(dirname "$CHECK_MODEL")" && pwd)/$(basename "$CHECK_MODEL")"
fi

SIGN=0
if [ -n "${STEMS_SIGN_IDENTITY:-}" ]; then SIGN=1; fi
NOTARIZE=0
if [ -n "${STEMS_NOTARY_KEY:-}" ] && [ -n "${STEMS_NOTARY_KEY_ID:-}" ] && [ -n "${STEMS_NOTARY_ISSUER:-}" ]; then
  NOTARIZE=1
  [ -f "$STEMS_NOTARY_KEY" ] || fail "STEMS_NOTARY_KEY $STEMS_NOTARY_KEY does not exist"
fi
if [ "$REQUIRE_SIGNING" = 1 ] && { [ "$SIGN" = 0 ] || [ "$NOTARIZE" = 0 ]; }; then
  fail "--require-signing: STEMS_SIGN_IDENTITY and the STEMS_NOTARY_* variables must all be set"
fi
if [ "$NOTARIZE" = 1 ] && [ "$SIGN" = 0 ]; then fail "notarizing needs STEMS_SIGN_IDENTITY too"; fi

command -v cmake >/dev/null || fail "cmake is not on PATH"
CTEST="$(dirname "$(command -v cmake)")/ctest"

echo "stems.cpp  $(git rev-parse --short HEAD)"
echo "ggml       $(git -C ggml rev-parse HEAD)"
echo "version    $VERSION"
echo "macOS      $(sw_vers -productVersion), deployment target $DEPLOYMENT_TARGET"
echo "xcode      $(xcodebuild -version | head -1)"
echo "signing    $([ "$SIGN" = 1 ] && echo "Developer ID" || echo "ad-hoc (local check only)")"
echo "notarize   $([ "$NOTARIZE" = 1 ] && echo yes || echo no)"

# --- build ------------------------------------------------------------------

cmake -S . -B "$BUILD_PATH" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
  -DGGML_NATIVE=OFF \
  -DGGML_BACKEND_DL=OFF \
  -DSTEMS_METAL=ON \
  -DSTEMS_CUDA=OFF \
  -DSTEMS_VULKAN=OFF \
  -DSTEMS_BUILD_TOOLS=ON \
  -DBUILD_TESTING=ON
cmake --build "$BUILD_PATH" --config Release --parallel "$JOBS"

if [ "$SKIP_TESTS" = 0 ]; then
  "$CTEST" --test-dir "$BUILD_PATH" -C Release --output-on-failure
fi

BIN="$BUILD_PATH/bin"
for tool in stems-server stems-split; do
  reported="$("$BIN/$tool" --version 2>/dev/null)" || fail "$tool --version failed"
  [ "v$reported" = "$VERSION" ] || fail "$tool reports $reported but the package is $VERSION; update project(VERSION) in CMakeLists.txt"
done

# --- stage ------------------------------------------------------------------

STAGE="$BUILD_PATH/package/macos"
rm -rf "$BUILD_PATH/package"
mkdir -p "$STAGE"

cp "$BIN/libstems.dylib" "$BIN/stems-server" "$BIN/stems-split" "$STAGE/"
cp LICENSE "$STAGE/"
cp src/libstems_v1.h "$STAGE/"
cp models.sh "$STAGE/"
cp docs/RUNTIME_RELEASE.md "$STAGE/RUNTIME_README.md"
cp docs/THIRD_PARTY_NOTICES.md "$STAGE/THIRD_PARTY_NOTICES.md"
cp ggml/LICENSE "$STAGE/LICENSE-ggml.txt"
cp vendor/cpp-httplib/LICENSE "$STAGE/LICENSE-cpp-httplib.txt"
cp vendor/yyjson/LICENSE "$STAGE/LICENSE-yyjson.txt"
chmod 755 "$STAGE/stems-server" "$STAGE/stems-split" "$STAGE/models.sh"

DIRTY=false
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then DIRTY=true; fi
cat > "$STAGE/BUILD-INFO.json" <<EOF
{
  "service": "stems",
  "version": "$VERSION",
  "commit": "$(git rev-parse HEAD)",
  "dirty": $DIRTY,
  "ggml_commit": "$(git -C ggml rev-parse HEAD)",
  "platform": "macos-arm64",
  "backends": ["metal"],
  "macos_deployment_target": "$DEPLOYMENT_TARGET",
  "xcode": "$(xcodebuild -version | head -1)",
  "signed": $([ "$SIGN" = 1 ] && echo true || echo false),
  "built_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# The dylib must be self-contained: one exported symbol, and nothing linked but the
# system. A stray @rpath dependency would work here and fail on a user's machine.
exports="$(nm -gU "$STAGE/libstems.dylib" | awk '{print $NF}')"
[ "$exports" = "_stems_get_api" ] || fail "libstems.dylib exports more than _stems_get_api: $exports"
for f in libstems.dylib stems-server stems-split; do
  lipo -archs "$STAGE/$f" | grep -qx arm64 || fail "$f is not arm64-only: $(lipo -archs "$STAGE/$f")"
  deps="$(otool -L "$STAGE/$f" | tail -n +2 | awk '{print $1}' | grep -v -e '^/usr/lib/' -e '^/System/Library/' -e '^@rpath/libstems.dylib$' || true)"
  [ -z "$deps" ] || fail "$f links outside the system: $deps"
done

# --- sign -------------------------------------------------------------------
#
# Hardened runtime with no entitlements: nothing here JITs or loads foreign code, and
# Metal compiles its embedded shaders without one. The secure timestamp is required
# for notarization.

if [ "$SIGN" = 1 ]; then
  for f in libstems.dylib stems-server stems-split; do
    codesign --force --options runtime --timestamp --sign "$STEMS_SIGN_IDENTITY" "$STAGE/$f"
    codesign --verify --strict --verbose=2 "$STAGE/$f"
  done
  codesign -dvv "$STAGE/libstems.dylib" 2>&1 | grep -E '^(Authority|TeamIdentifier|Timestamp)=' | head -3
fi

# --- check the staged package from outside it --------------------------------
#
# From a scratch folder that holds nothing of the package, the way a DAW runs: the
# executables must start with only what the zip ships, and libstems.dylib, dlopened by
# absolute path from a foreign executable, must separate audio. Runs after signing, so
# it is the signed binaries that are checked.

SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/stems-package-check.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
cp "$BIN/stems-abi-test" "$SCRATCH/"
(
  cd "$SCRATCH"
  reported="$("$STAGE/stems-server" --version 2>/dev/null)"
  [ "v$reported" = "$VERSION" ] || fail "staged stems-server reports $reported"
  "$STAGE/stems-server" --props --models-dir "$SCRATCH" 2>/dev/null > props.json
  gpu="$(/usr/bin/python3 -c 'import json,sys; d=json.load(open("props.json"))["devices"]; print(next((x["description"] for x in d if x["backend"]=="MTL"), ""))')"
  [ -n "$gpu" ] || fail "staged stems-server --props lists no Metal device"
  echo "Metal device: $gpu"
  if [ -n "$CHECK_MODEL" ]; then
    case "$gpu" in
      *Paravirtual*) echo "paravirtualized GPU: separating on the CPU instead"; export STEMS_DEVICE=cpu ;;
    esac
    ./stems-abi-test "$STAGE/libstems.dylib" "${VERSION#v}" "$CHECK_MODEL" 2>/dev/null
  else
    ./stems-abi-test "$STAGE/libstems.dylib" "${VERSION#v}" 2>/dev/null
  fi
)
echo "staged package checked from outside its folder"

# --- zip, notarize, checksum --------------------------------------------------

mkdir -p "$OUT_PATH"
for f in "$OUT_PATH"/*; do
  [ -e "$f" ] || continue
  case "$(basename "$f")" in
    stems-v*-macos-arm64.zip|SHA256SUMS-macos) rm -f "$f" ;;
    *) fail "$OUT_PATH holds files that are not macOS package outputs: $(basename "$f")" ;;
  esac
done

ZIP_NAME="stems-$VERSION-macos-arm64.zip"
ZIP="$OUT_PATH/$ZIP_NAME"
# Flat at the root, like the Windows zips. ditto keeps the signatures and modes intact.
ditto -c -k --norsrc "$STAGE" "$ZIP"

if [ "$NOTARIZE" = 1 ]; then
  echo "notarizing $ZIP_NAME"
  result="$(xcrun notarytool submit "$ZIP" --key "$STEMS_NOTARY_KEY" --key-id "$STEMS_NOTARY_KEY_ID" \
            --issuer "$STEMS_NOTARY_ISSUER" --wait --timeout 30m --output-format json)" || true
  status="$(printf '%s' "$result" | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin).get("status",""))' 2>/dev/null || true)"
  id="$(printf '%s' "$result" | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))' 2>/dev/null || true)"
  echo "notarization $id: ${status:-no status}"
  if [ "$status" != "Accepted" ]; then
    echo "$result" >&2
    [ -n "$id" ] && xcrun notarytool log "$id" --key "$STEMS_NOTARY_KEY" --key-id "$STEMS_NOTARY_KEY_ID" \
                      --issuer "$STEMS_NOTARY_ISSUER" >&2 || true
    fail "notarization was not accepted"
  fi
fi

( cd "$OUT_PATH" && shasum -a 256 "$ZIP_NAME" > SHA256SUMS-macos )
printf '%-36s %8s MB  %s\n' "$ZIP_NAME" "$(du -m "$ZIP" | cut -f1)" "$(cut -d' ' -f1 "$OUT_PATH/SHA256SUMS-macos")"
echo "package -> $OUT_PATH"

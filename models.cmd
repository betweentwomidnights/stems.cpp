@echo off
setlocal enabledelayedexpansion
rem Download stems.cpp GGUFs from Hugging Face (public repos) with curl.exe. No Python needed.
rem Usage: models.cmd [--encoding f32^|f16] [--namespace ORG] [--out DIR] [--dry-run] [MODEL ...^|all]
rem   MODEL: htdemucs (default), htdemucs_6s, htdemucs_ft, mel_band_roformer_kim, bs_roformer_viperx_317,
rem          uvr_denoise, uvr_denoise_lite, uvr_deecho_normal, uvr_deecho_aggressive, uvr_deecho_dereverb
rem   defaults: htdemucs, F32, into .\models
rem Same behaviour as models.sh; see docs\DISTRIBUTION.md. F16 is published only where it was
rem measured to hold up; asking for it elsewhere fetches F32. bs_roformer_viperx_317 and the uvr_*
rem models have no stated upstream license: read their model cards before redistributing them.

set "ENCODING=f32"
set "NAMESPACE=thepatch"
set "OUT=models"
set "DRY_RUN=0"
set "MODELS="
rem Models with a published F16, space-delimited.
set "F16_PUBLISHED= htdemucs htdemucs_6s htdemucs_ft mel_band_roformer_kim "

:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--encoding"  ( set "ENCODING=%~2" & shift & shift & goto parse )
if /I "%~1"=="--namespace" ( set "NAMESPACE=%~2" & shift & shift & goto parse )
if /I "%~1"=="--out"       ( set "OUT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--dry-run"   ( set "DRY_RUN=1" & shift & goto parse )
if /I "%~1"=="-h"          goto help
if /I "%~1"=="--help"      goto help
set "ARG=%~1"
if "!ARG:~0,1!"=="-" ( echo unknown option: %~1 & exit /b 1 )
set "MODELS=!MODELS! %~1"
shift
goto parse
:parsed

if "%MODELS%"=="" set "MODELS= htdemucs"
if /I "%MODELS%"==" all" set "MODELS= htdemucs htdemucs_6s htdemucs_ft mel_band_roformer_kim bs_roformer_viperx_317 uvr_denoise uvr_denoise_lite uvr_deecho_normal uvr_deecho_aggressive uvr_deecho_dereverb"

if /I "%ENCODING%"=="f32" ( set "ENC=F32" & goto encoding_ok )
if /I "%ENCODING%"=="f16" ( set "ENC=F16" & goto encoding_ok )
echo unknown --encoding '%ENCODING%' ^(expected f32^|f16^) & exit /b 2
:encoding_ok

if not exist "%OUT%" mkdir "%OUT%"
for %%M in (%MODELS%) do (
  set "M=%%M"
  set "REPO=" & set "LABEL="
  if /I "!M!"=="htdemucs"               ( set "REPO=htdemucs-GGUF" & set "LABEL=42M" )
  if /I "!M!"=="htdemucs_6s"            ( set "REPO=htdemucs-GGUF" & set "LABEL=27M" )
  if /I "!M!"=="htdemucs_ft"            ( set "REPO=htdemucs-GGUF" & set "LABEL=4x42M" )
  if /I "!M!"=="mel_band_roformer_kim"  ( set "REPO=mel-band-roformer-kim-GGUF" & set "LABEL=0.2B" )
  if /I "!M!"=="bs_roformer_viperx_317" ( set "REPO=bs-roformer-viperx-317-GGUF" & set "LABEL=0.2B" )
  if /I "!M!"=="uvr_denoise"            ( set "REPO=uvr-vr-GGUF" & set "LABEL=32M" )
  if /I "!M!"=="uvr_denoise_lite"       ( set "REPO=uvr-vr-GGUF" & set "LABEL=4M" )
  if /I "!M!"=="uvr_deecho_normal"      ( set "REPO=uvr-vr-GGUF" & set "LABEL=32M" )
  if /I "!M!"=="uvr_deecho_aggressive"  ( set "REPO=uvr-vr-GGUF" & set "LABEL=32M" )
  if /I "!M!"=="uvr_deecho_dereverb"    ( set "REPO=uvr-vr-GGUF" & set "LABEL=56M" )
  if "!REPO!"=="" ( echo unknown model: !M! ^(try --help^) & exit /b 1 )
  set "E=%ENC%"
  if "!E!"=="F16" (
    set "CHECK=!F16_PUBLISHED: %%M =!"
    if "!CHECK!"=="!F16_PUBLISHED!" ( echo [note] !M! has no published F16; fetching F32 & set "E=F32" )
  )
  set "FILE=!M!-!LABEL!-v1.0-!E!.gguf"
  set "URL=https://huggingface.co/%NAMESPACE%/!REPO!/resolve/main/!FILE!"
  if "%DRY_RUN%"=="1" (
    echo [plan] !URL! -^> %OUT%\!FILE!
  ) else if exist "%OUT%\!FILE!" (
    echo [skip] !FILE!
  ) else (
    if exist "%OUT%\!FILE!.part" ( echo [resume] !FILE! ) else ( echo [download] %NAMESPACE%/!REPO!/!FILE! )
    curl.exe -fL --retry 3 --continue-at - -o "%OUT%\!FILE!.part" "!URL!" || exit /b 1
    move /y "%OUT%\!FILE!.part" "%OUT%\!FILE!" >nul
  )
)
echo [done]%MODELS% -^> %OUT%
exit /b 0

:help
for /f "skip=2 tokens=1,* delims= " %%A in ('findstr /b "rem" "%~f0"') do (
  if "%%A"=="rem" echo(%%B
)
exit /b 0

stems.cpp standalone Windows package
====================================

This folder contains stems-server.exe, stems-split.exe, stems.dll (the C ABI
library, with libstems_v1.h), the CPU backend variants, the CUDA and Vulkan
backends, and the CUDA runtime. Model weights are downloaded separately.
BUILD-INFO.json records the source commit and toolchain used to build it.

1. Download a model from this folder. Examples:

     models.cmd                                  (htdemucs, F32)
     models.cmd --encoding f16 htdemucs_6s
     models.cmd all

   models.cmd --help lists the models. Model licenses and access terms remain
   with their publishers.

2. Separate a song from the command line:

     stems-split.exe --model models\htdemucs-42M-v1.0-F32.gguf --input song.wav --out stems

   or run the HTTP service and send it base64 WAVs:

     stems-server.exe --models-dir models --port 8010

   Both answer --version, and stems-server --props lists the ggml devices and
   which model files it found, without loading a model.

The best GPU is used when one is available, falling back to the CPU; pass
--device cpu to force the CPU. CUDA works when the NVIDIA driver supports this
toolkit; Vulkan is available for supported GPUs. See RUNTIME_README.md.

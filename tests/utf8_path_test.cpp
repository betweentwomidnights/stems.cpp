// utf8_path_test: a GGUF in a folder whose name is not ASCII loads, metadata and tensor data.
// Hosts pass UTF-8 paths (juce::File::toRawUTF8()), and gary4juce keeps models under the user's
// Documents folder, so a user named José puts "é" in every model path.
#include "gguf_model.h"

#include <cstdio>
#include <cstring>
#include <string>
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace {

// Create (or reuse) the directory, given as UTF-8.
bool make_dir_utf8(const std::string& dir) {
#ifdef _WIN32
    std::wstring w(MultiByteToWideChar(CP_UTF8, 0, dir.c_str(), -1, nullptr, 0), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, dir.c_str(), -1, &w[0], (int)w.size());
    return CreateDirectoryW(w.c_str(), nullptr) || GetLastError() == ERROR_ALREADY_EXISTS;
#else
    return mkdir(dir.c_str(), 0755) == 0 || access(dir.c_str(), F_OK) == 0;
#endif
}

} // namespace

int main(int argc, char** argv) {
    // The parent directory comes from CTest (the build tree), which is plain ASCII.
    const std::string parent = argc > 1 ? argv[1] : ".";
    const std::string dir = parent + "/t\xc3\xa9st-jos\xc3\xa9";
    const std::string path = dir + "/tiny.gguf";
    if (!make_dir_utf8(dir)) { printf("FAIL: cannot create %s\n", dir.c_str()); return 1; }

    const float values[4] = {1.0f, -2.5f, 3.25f, 0.125f};
    {
        ggml_init_params ip = { /*mem_size=*/1024 * 1024, /*mem_buffer=*/nullptr, /*no_alloc=*/false };
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_set_name(t, "w");
        std::memcpy(t->data, values, sizeof values);
        gguf_context* g = gguf_init_empty();
        gguf_set_val_str(g, "general.architecture", "test");
        gguf_add_tensor(g, t);
        const bool ok = gguf_write_to_file(g, path.c_str(), false);   // ggml_fopen: UTF-8 aware
        gguf_free(g);
        ggml_free(ctx);
        if (!ok) { printf("FAIL: cannot write %s\n", path.c_str()); return 1; }
    }

    try {
        st::GgufModel m = st::load_gguf(path.c_str());
        float got[4] = {};
        ggml_backend_tensor_get(m.get("w"), got, 0, sizeof got);
        const bool same = std::memcmp(got, values, sizeof values) == 0;
        printf("  load_gguf from a non-ASCII folder: %s\n", same ? "ok" : "FAIL (wrong data)");
        return same ? 0 : 1;
    } catch (const std::exception& e) {
        printf("  load_gguf from a non-ASCII folder: FAIL (%s)\n", e.what());
        return 1;
    }
}

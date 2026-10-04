#include "gguf_model.h"

#include <mutex>
#include <string>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace st {
namespace {

// The folder holding the module (DLL, shared object or executable) this function lives in,
// UTF-8, or empty if it cannot be found.
std::string this_module_dir() {
    static const char anchor = 0;
#ifdef _WIN32
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)&anchor, &module))
        return {};
    std::wstring path(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(module, &path[0], (DWORD)path.size());
        if (n == 0) return {};
        if (n < path.size()) { path.resize(n); break; }
        path.resize(path.size() * 2);
    }
    const size_t slash = path.find_last_of(L"\\/");
    if (slash == std::wstring::npos) return {};
    path.resize(slash);
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, path.c_str(), (int)path.size(), nullptr, 0, nullptr, nullptr);
    std::string out((size_t)bytes, '\0');
    WideCharToMultiByte(CP_UTF8, 0, path.c_str(), (int)path.size(), &out[0], bytes, nullptr, nullptr);
    return out;
#else
    Dl_info info;
    if (!dladdr((const void*)&anchor, &info) || !info.dli_fname) return {};
    const std::string path = info.dli_fname;
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
#endif
}

} // namespace

void load_dynamic_backends_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        const std::string dir = this_module_dir();
        if (!dir.empty()) ggml_backend_load_all_from_path(dir.c_str());
        if (ggml_backend_dev_count() == 0) ggml_backend_load_all();
    });
}

} // namespace st

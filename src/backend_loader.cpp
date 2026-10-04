#include "gguf_model.h"

#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif
#ifdef __APPLE__
#  include <sys/sysctl.h>
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

int physical_cores() {
#if defined(__APPLE__)
    // Performance cores on Apple Silicon (perflevel0); every physical core on Intel Macs.
    for (const char* key : {"hw.perflevel0.physicalcpu", "hw.physicalcpu"}) {
        int n = 0;
        size_t size = sizeof(n);
        if (sysctlbyname(key, &n, &size, nullptr, 0) == 0 && n > 0) return n;
    }
    return 0;
#elif defined(_WIN32)
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes == 0) return 0;
    std::vector<char> buf(bytes);
    auto* info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)buf.data();
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &bytes)) return 0;
    int n = 0;
    for (DWORD off = 0; off < bytes; n++)
        off += ((SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(buf.data() + off))->Size;
    return n;
#else
    // One entry per core: the logical CPUs that share it list the same siblings.
    std::set<std::string> cores;
    const unsigned logical = std::thread::hardware_concurrency();
    for (unsigned i = 0; i < logical; i++) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/thread_siblings_list");
        std::string siblings;
        if (f && std::getline(f, siblings)) cores.insert(siblings);
    }
    return cores.empty() ? (int)logical : (int)cores.size();
#endif
}

} // namespace

int default_cpu_threads() {
    static const int n = physical_cores();
    return n;
}

void load_dynamic_backends_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        const std::string dir = this_module_dir();
        if (!dir.empty()) ggml_backend_load_all_from_path(dir.c_str());
        if (ggml_backend_dev_count() == 0) ggml_backend_load_all();
    });
}

} // namespace st

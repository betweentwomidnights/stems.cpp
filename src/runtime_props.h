// runtime_props.h -- what gary4local reads from `stems-server --props` (and GET /props) before
// it installs a model or binds a port: the version and the ggml devices this package can use.
// Same shape as sa3.cpp's, so one installer check covers every native runtime.
#pragma once

#include "gguf_model.h"

#include <string>

#ifndef STEMS_VERSION_STRING
#define STEMS_VERSION_STRING "0.0.0"
#endif

namespace st {

inline const char* runtime_version() { return STEMS_VERSION_STRING; }

inline std::string runtime_json_quote(const char* input) {
    std::string out = "\"";
    for (const unsigned char* p = (const unsigned char*)(input ? input : ""); *p; ++p) {
        const unsigned char c = *p;
        if (c == '"' || c == '\\') { out += '\\'; out += (char)c; }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) {
            constexpr char hex[] = "0123456789abcdef";
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else out += (char)c;
    }
    return out + '"';
}

// The registry name (CUDA/Vulkan/CPU) goes in "backend", not the numbered device name.
// `extra` is appended verbatim as further top-level fields (",\"key\":value...").
inline std::string runtime_props_json(const char* service, const std::string& extra = {}) {
    load_dynamic_backends_once();
    std::string body = "{\"success\":true,\"service\":" + runtime_json_quote(service) +
        ",\"version\":" + runtime_json_quote(runtime_version()) + ",\"devices\":[";
    bool first = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const auto kind = ggml_backend_dev_type(dev);
        if (kind != GGML_BACKEND_DEVICE_TYPE_CPU && kind != GGML_BACKEND_DEVICE_TYPE_GPU &&
            kind != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        const ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        if (!first) body += ',';
        first = false;
        body += "{\"name\":" + runtime_json_quote(ggml_backend_dev_name(dev)) +
            ",\"description\":" + runtime_json_quote(ggml_backend_dev_description(dev)) +
            ",\"backend\":" + runtime_json_quote(reg ? ggml_backend_reg_name(reg) : "") +
            ",\"type\":" + runtime_json_quote(kind == GGML_BACKEND_DEVICE_TYPE_CPU ? "cpu" :
                kind == GGML_BACKEND_DEVICE_TYPE_IGPU ? "integrated_gpu" : "gpu") +
            ",\"memory_free_bytes\":" + std::to_string(free_bytes) +
            ",\"memory_total_bytes\":" + std::to_string(total_bytes) + "}";
    }
    return body + "]" + extra + "}";
}

} // namespace st

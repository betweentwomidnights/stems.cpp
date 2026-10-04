/* abi_test: the libstems V1 contract a host (gary4juce, an iplug2 plugin) relies on.
 *
 *   abi_test LIBRARY VERSION            no model: the CTest
 *   abi_test LIBRARY VERSION MODEL      also separate one second of noise with MODEL
 *
 * LIBRARY (stems.dll / libstems.so / libstems.dylib, an absolute path) is loaded at run time
 * and only stems_get_api is resolved, exactly as a plugin does; nothing links against it. Run
 * from a folder without ggml in it, this is the DAW case ci/package-windows.ps1 checks on a
 * staged package: stems.dll must find its own ggml backends beside itself.
 * Pure C on purpose, like stems-libtest: the public header has to stay C-clean.
 */
#include "libstems_v1.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

typedef const stems_api_v1* (STEMS_CALL *get_api_fn)(uint32_t);

static int failures = 0;

static void check(int ok, const char* what) {
    printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

static get_api_fn load(const char* library) {
#ifdef _WIN32
    /* LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR finds ggml.dll beside stems.dll, as a plugin loads it;
     * it wants a fully qualified path with backslashes (CMake's $<TARGET_FILE> has '/'). */
    char path[MAX_PATH];
    snprintf(path, sizeof path, "%s", library);
    for (char* p = path; *p; p++) if (*p == '/') *p = '\\';
    HMODULE m = LoadLibraryExA(path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!m) { fprintf(stderr, "LoadLibraryEx(%s) failed: %lu\n", path, GetLastError()); return NULL; }
    return (get_api_fn)GetProcAddress(m, "stems_get_api");
#else
    void* m = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (!m) { fprintf(stderr, "dlopen(%s) failed: %s\n", library, dlerror()); return NULL; }
    return (get_api_fn)dlsym(m, "stems_get_api");
#endif
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: abi_test LIBRARY VERSION [MODEL]\n");
        return 2;
    }
    const char* version = argv[2];
    const char* model = argc > 3 ? argv[3] : NULL;
    get_api_fn get_api = load(argv[1]);
    if (!get_api) { printf("FAIL: no stems_get_api in %s\n", argv[1]); return 1; }

    printf("api table\n");
    const stems_api_v1* api = get_api(STEMS_ABI_VERSION_1);
    check(api != NULL, "stems_get_api(1) returns the V1 table");
    if (!api) return 1;
    check(get_api(2) == NULL, "stems_get_api(2) returns NULL");
    check(api->abi_version == STEMS_ABI_VERSION_1 && api->size >= STEMS_API_V1_MIN_SIZE,
          "abi_version 1, size covers the frozen V1 table");
    const char* rv = api->runtime_version();
    printf("  runtime_version: %s\n", rv ? rv : "(null)");
    check(rv && strstr(rv, version) != NULL, "runtime_version names the project version");

    printf("initializers and argument checks\n");
    stems_error_v1 err;
    memset(&err, 0, sizeof err); err.size = sizeof err; api->error_init(&err);
    check(err.code == STEMS_STATUS_OK_V1 && err.message[0] == '\0', "error_init clears the error");

    stems_context_config_v1 cfg;
    memset(&cfg, 0, sizeof cfg); cfg.size = sizeof cfg; api->context_config_init(&cfg);
    stems_context* ctx = NULL;
    check(api->context_create(NULL, &ctx, &err) == STEMS_STATUS_INVALID_ARGUMENT_V1 && ctx == NULL,
          "context_create(NULL config) is INVALID_ARGUMENT");
    stems_context_config_v1 truncated = cfg;   /* not "small": <windows.h> defines that */
    truncated.size = 4;
    check(api->context_create(&truncated, &ctx, &err) == STEMS_STATUS_INVALID_ARGUMENT_V1 && ctx == NULL,
          "context_create with a too-small size is INVALID_ARGUMENT");
    cfg.model_path = "this/model/does/not/exist.gguf";
    stems_status_v1 st = api->context_create(&cfg, &ctx, &err);
    check(st != STEMS_STATUS_OK_V1 && ctx == NULL && err.message[0] != '\0',
          "a missing model fails with a message, no context");
    printf("    (status %d: %s)\n", (int)st, err.message);

    stems_request_v1 req;
    memset(&req, 0, sizeof req); req.size = sizeof req; api->request_init(&req);
    check(req.overlap < 0.0f && req.shifts == 0, "request_init: model overlap, no shifts");
    stems_result_v1 res;
    memset(&res, 0, sizeof res); res.size = sizeof res; api->result_init(&res);
    check(api->separate(NULL, &req, &res, &err) == STEMS_STATUS_INVALID_ARGUMENT_V1,
          "separate(NULL context) is INVALID_ARGUMENT");
    api->result_free(&res);
    api->result_free(&res);
    check(1, "result_free is safe on an empty result, twice");

    if (model) {
        printf("separation with %s\n", model);
        cfg.model_path = model;
        st = api->context_create(&cfg, &ctx, &err);
        check(st == STEMS_STATUS_OK_V1 && ctx != NULL, "context_create with the model");
        if (st != STEMS_STATUS_OK_V1) { printf("    (status %d: %s)\n", (int)st, err.message); return 1; }

        stems_model_info_v1 info;
        memset(&info, 0, sizeof info); info.size = sizeof info; api->model_info_init(&info);
        check(api->model_info(ctx, &info, &err) == STEMS_STATUS_OK_V1 && info.n_sources >= 1,
              "model_info");
        printf("  model %s on %s, %u sources\n", info.name, info.backend, info.n_sources);

        const uint64_t n = 44100;
        float* noise = (float*)malloc(sizeof(float) * 2 * n);
        srand(1);
        for (uint64_t i = 0; i < 2 * n; i++) noise[i] = 0.2f * ((float)rand() / RAND_MAX - 0.5f);
        req.input.samples = noise;
        req.input.n_samples = n;
        req.input.n_channels = 2;
        req.input.sample_rate = 44100;
        req.input.layout = STEMS_AUDIO_PLANAR_V1;
        st = api->separate(ctx, &req, &res, &err);
        check(st == STEMS_STATUS_OK_V1 && res.n_sources == info.n_sources && res.n_samples == n &&
              res.n_channels == 2, "separate returns every stem at the input length");
        if (st == STEMS_STATUS_OK_V1) {
            int finite = 1;
            for (uint64_t i = 0; i < (uint64_t)res.n_sources * res.n_channels * res.n_samples; i++)
                if (!isfinite(res.samples[i])) { finite = 0; break; }
            check(finite, "every output sample is finite");
        } else {
            printf("    (status %d: %s)\n", (int)st, err.message);
        }
        api->result_free(&res);
        api->context_destroy(ctx);
        free(noise);
    }

    printf(failures ? "FAIL: %d check(s)\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}

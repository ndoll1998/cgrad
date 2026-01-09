#include "cgrad_backend.h"
#include "backends/cgrad_tensor_f32_cpu.h"

static int  backend_init(void* t, const uint32_t* shape) {
    return cgrad_tensor_f32_init((cgrad_tensor_f32*)t, shape);
}
static int  backend_fill_rand(void* t) {
    return cgrad_tensor_f32_fill_rand((cgrad_tensor_f32*)t);
}
static int  backend_gemm(void* a, void* b, void* c) {
    return cgrad_tensor_f32_gemm((cgrad_tensor_f32*)a, (cgrad_tensor_f32*)b, (cgrad_tensor_f32*)c);
}
static void backend_free(void* t) {
    cgrad_tensor_f32_free((cgrad_tensor_f32*)t);
}
static void backend_print(const void* t) {
    cgrad_tensor_f32_print((const cgrad_tensor_f32*)t);
}
static void backend_transpose(void* t, const uint32_t* perm) {
    cgrad_tensor_f32_transpose((cgrad_tensor_f32*)t, perm);
}

static cgrad_backend cgrad_backend_cpu = {
    .type = CGRAD_BACKEND_F32_CPU,
    .init      = backend_init,
    .fill_rand = backend_fill_rand,
    .gemm      = backend_gemm,
    .free      = backend_free,
    .print     = backend_print,
    .transpose = backend_transpose,
};

// Backend registry
cgrad_backend* cgrad_get_backend(cgrad_backend_type type) {
    switch (type) {
        case CGRAD_BACKEND_F32_CPU:
            return &cgrad_backend_cpu;
        // Add more backends here
        default:
            return NULL;
    }
}

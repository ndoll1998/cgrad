#include "cgrad_backend.h"
#include "backends/cgrad_tensor_f32_cpu.h"

static int  cgrad_tensor_f32_cpu_backend_init(void* t, const uint32_t* shape) {
    return cgrad_tensor_f32_cpu_init((cgrad_tensor_f32*)t, shape);
}
static int  cgrad_tensor_f32_cpu_backend_fill_rand(void* t) {
    return cgrad_tensor_f32_cpu_fill_rand((cgrad_tensor_f32*)t);
}
static int  cgrad_tensor_f32_cpu_backend_gemm(void* a, void* b, void* c) {
    return cgrad_tensor_f32_cpu_gemm((cgrad_tensor_f32*)a, (cgrad_tensor_f32*)b, (cgrad_tensor_f32*)c);
}
static void cgrad_tensor_f32_cpu_backend_free(void* t) {
    cgrad_tensor_f32_cpu_free((cgrad_tensor_f32*)t);
}
static void cgrad_tensor_f32_cpu_backend_print(const void* t) {
    cgrad_tensor_f32_cpu_print((const cgrad_tensor_f32*)t);
}
static void cgrad_tensor_f32_cpu_backend_transpose(void* t, const uint32_t* perm) {
    cgrad_tensor_f32_cpu_transpose((cgrad_tensor_f32*)t, perm);
}

static cgrad_backend cgrad_backend_cpu = {
    .type = CGRAD_BACKEND_F32_CPU,
    .init      = cgrad_tensor_f32_cpu_backend_init,
    .fill_rand = cgrad_tensor_f32_cpu_backend_fill_rand,
    .gemm      = cgrad_tensor_f32_cpu_backend_gemm,
    .free      = cgrad_tensor_f32_cpu_backend_free,
    .print     = cgrad_tensor_f32_cpu_backend_print,
    .transpose = cgrad_tensor_f32_cpu_backend_transpose,
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

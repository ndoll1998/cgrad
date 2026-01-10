#include "cgrad_backend.h"
#include "backends/cgrad_tensor_f32_cpu.h"
#include <stdlib.h>

static int  cgrad_tensor_f32_cpu_backend_tensor_init(void* t, const uint32_t* shape) {
    return cgrad_tensor_f32_cpu_init((cgrad_tensor_f32_cpu*)t, shape);
}
static int  cgrad_tensor_f32_cpu_backend_tensor_fill_rand(void* t) {
    return cgrad_tensor_f32_cpu_fill_rand((cgrad_tensor_f32_cpu*)t);
}
static int  cgrad_tensor_f32_cpu_backend_tensor_gemm(void* a, void* b, void* c) {
    return cgrad_tensor_f32_cpu_gemm((cgrad_tensor_f32_cpu*)a, (cgrad_tensor_f32_cpu*)b, (cgrad_tensor_f32_cpu*)c);
}
static void cgrad_tensor_f32_cpu_backend_tensor_free(void* t) {
    cgrad_tensor_f32_cpu_free((cgrad_tensor_f32_cpu*)t);
}
static void cgrad_tensor_f32_cpu_backend_tensor_print(const void* t) {
    cgrad_tensor_f32_cpu_print((const cgrad_tensor_f32_cpu*)t);
}
static void cgrad_tensor_f32_cpu_backend_tensor_transpose(void* t, const uint32_t* perm) {
    cgrad_tensor_f32_cpu_transpose((cgrad_tensor_f32_cpu*)t, perm);
}

// Allocator for cgrad_tensor_f32_cpu handle
static void* alloc_tensor_f32_cpu_handle(void) {
    return malloc(sizeof(struct cgrad_tensor_f32_cpu));
}

static cgrad_backend cgrad_backend_cpu = {
    .type = CGRAD_BACKEND_F32_CPU,
    .alloc_tensor_handle = alloc_tensor_f32_cpu_handle,
    .tensor_init      = cgrad_tensor_f32_cpu_backend_tensor_init,
    .tensor_fill_rand = cgrad_tensor_f32_cpu_backend_tensor_fill_rand,
    .tensor_gemm      = cgrad_tensor_f32_cpu_backend_tensor_gemm,
    .tensor_free      = cgrad_tensor_f32_cpu_backend_tensor_free,
    .tensor_print     = cgrad_tensor_f32_cpu_backend_tensor_print,
    .tensor_transpose = cgrad_tensor_f32_cpu_backend_tensor_transpose,
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

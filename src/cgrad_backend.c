#include "cgrad_backend.h"
#include "backends/cgrad_tensor_f32_cpu.h"
#include <stdlib.h>

static void* _alloc_tensor_f32_cpu_handle(void) {
    return malloc(sizeof(struct cgrad_tensor_f32_cpu));
}
static int _cgrad_tensor_f32_cpu_backend_tensor_init(void* t, const uint32_t* shape) {
    return cgrad_tensor_f32_cpu_init((cgrad_tensor_f32_cpu*)t, shape);
}
static int _cgrad_tensor_f32_cpu_backend_tensor_fill_rand(void* t) {
    return cgrad_tensor_f32_cpu_fill_rand((cgrad_tensor_f32_cpu*)t);
}
static int _cgrad_tensor_f32_cpu_backend_tensor_shallow_copy(const void* src, void* dst) {
    return cgrad_tensor_f32_cpu_shallow_copy(
        (const cgrad_tensor_f32_cpu*)src,
        (cgrad_tensor_f32_cpu*)dst
    );
}
static void _cgrad_tensor_f32_cpu_backend_tensor_free(void* t) {
    cgrad_tensor_f32_cpu_free((cgrad_tensor_f32_cpu*)t);
}
static int _cgrad_tensor_f32_cpu_backend_tensor_add(void* a, void* b, void* c) {
    return cgrad_tensor_f32_cpu_add(
        (const cgrad_tensor_f32_cpu*)a,
        (const cgrad_tensor_f32_cpu*)b,
        (cgrad_tensor_f32_cpu*)c
    );
}
static int _cgrad_tensor_f32_cpu_backend_tensor_gemm(void* a, void* b, void* c) {
    return cgrad_tensor_f32_cpu_gemm((cgrad_tensor_f32_cpu*)a, (cgrad_tensor_f32_cpu*)b, (cgrad_tensor_f32_cpu*)c);
}
static cgrad_tensor_layout* _cgrad_tensor_f32_cpu_backend_tensor_get_layout(void* t) {
    return cgrad_tensor_f32_cpu_get_layout((cgrad_tensor_f32_cpu*)t);
}
static void _cgrad_tensor_f32_cpu_backend_tensor_print(const void* t) {
    cgrad_tensor_f32_cpu_print((const cgrad_tensor_f32_cpu*)t);
}
static void _cgrad_tensor_f32_cpu_backend_tensor_transpose(void* t, const uint32_t* perm) {
    cgrad_tensor_f32_cpu_transpose((cgrad_tensor_f32_cpu*)t, perm);
}

static cgrad_backend cgrad_backend_cpu = {
    .type = CGRAD_BACKEND_F32_CPU,
    .alloc_tensor_handle = _alloc_tensor_f32_cpu_handle,
    .tensor_init      = _cgrad_tensor_f32_cpu_backend_tensor_init,
    .tensor_fill_rand = _cgrad_tensor_f32_cpu_backend_tensor_fill_rand,
    .tensor_shallow_copy = _cgrad_tensor_f32_cpu_backend_tensor_shallow_copy,
    .tensor_free      = _cgrad_tensor_f32_cpu_backend_tensor_free,
    .tensor_add       = _cgrad_tensor_f32_cpu_backend_tensor_add,
    .tensor_gemm      = _cgrad_tensor_f32_cpu_backend_tensor_gemm,
    .tensor_get_layout   = _cgrad_tensor_f32_cpu_backend_tensor_get_layout,
    .tensor_print     = _cgrad_tensor_f32_cpu_backend_tensor_print,
    .tensor_transpose = _cgrad_tensor_f32_cpu_backend_tensor_transpose,
};

cgrad_backend* cgrad_get_backend(cgrad_backend_type type) {
    switch (type) {
        case CGRAD_BACKEND_F32_CPU:
            return &cgrad_backend_cpu;
        // Add more backends here
        default:
            return NULL;
    }
}

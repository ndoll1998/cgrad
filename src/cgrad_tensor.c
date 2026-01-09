#include "cgrad_tensor.h"
#include "backends/cgrad_tensor_f32_cpu.h"
#include <stdlib.h>
#include <string.h>

int cgrad_tensor_init(cgrad_tensor* t, const uint32_t* shape, cgrad_backend_type backend_type) {
    if (!t || !shape) return -1;
    t->backend = cgrad_get_backend(backend_type);
    if (!t->backend) return -1;

    // For now, only support f32 tensors and CPU backend
    cgrad_tensor_f32* handle = (cgrad_tensor_f32*)malloc(sizeof(cgrad_tensor_f32));
    if (!handle) return -1;
    if (t->backend->init(handle, shape)) {
        free(handle);
        return -1;
    }
    t->handle = handle;
    return 0;
}

void cgrad_tensor_free(cgrad_tensor* t) {
    if (!t || !t->backend || !t->handle) return;
    t->backend->free(t->handle);
    free(t->handle);
    t->handle = NULL;
}

int cgrad_tensor_fill_rand(cgrad_tensor* t) {
    if (!t || !t->backend || !t->handle) return -1;
    return t->backend->fill_rand(t->handle);
}

int cgrad_tensor_gemm(cgrad_tensor* a, cgrad_tensor* b, cgrad_tensor* c) {
    if (!a || !b || !c) return -1;
    if (!a->backend || !b->backend || !c->backend) return -1;
    if (a->backend != b->backend || a->backend != c->backend) return -1;
    return a->backend->gemm(a->handle, b->handle, c->handle);
}

void cgrad_tensor_print(const cgrad_tensor* t) {
    if (!t || !t->backend || !t->handle) return;
    t->backend->print(t->handle);
}

void cgrad_tensor_transpose(cgrad_tensor* t, const uint32_t* perm) {
    if (!t || !t->backend || !t->handle) return;
    t->backend->transpose(t->handle, perm);
}
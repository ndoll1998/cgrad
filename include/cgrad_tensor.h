#ifndef CGRAD_TENSOR_BASE_H
#define CGRAD_TENSOR_BASE_H

#include "cgrad_backend.h"
#include <stdint.h>
#include <stddef.h>

// High-level tensor object supporting multiple backends
typedef struct cgrad_tensor {
    cgrad_backend* backend; // Pointer to backend ops
    void* handle;           // Backend-specific tensor object (e.g., cgrad_tensor_f32*)
} cgrad_tensor;

// API for high-level tensor
int cgrad_tensor_init(cgrad_tensor* t, const uint32_t* shape, cgrad_backend_type backend_type);
void cgrad_tensor_free(cgrad_tensor* t);
int cgrad_tensor_fill_rand(cgrad_tensor* t);
int cgrad_tensor_gemm(const cgrad_tensor* a, const cgrad_tensor* b, cgrad_tensor* c);
int cgrad_tensor_add(cgrad_tensor* a, cgrad_tensor* b, cgrad_tensor* c);
void cgrad_tensor_print(const cgrad_tensor* t);
void cgrad_tensor_transpose(cgrad_tensor* t, const uint32_t* perm);

// Backend registry (for internal use)
cgrad_backend* cgrad_get_backend(cgrad_backend_type type);

#endif // CGRAD_TENSOR_BASE_H
#ifndef CGRAD_BACKEND_H
#define CGRAD_BACKEND_H

#include <stdint.h>
#include <stddef.h>
#include "cgrad_layout.h"

// Enum for backend types
typedef enum {
    CGRAD_BACKEND_F32_CPU = 0,
    // Add more backends here (e.g., CGRAD_BACKEND_F64_CPU, CGRAD_BACKEND_CUDA, ...)
} cgrad_backend_type;

typedef struct cgrad_backend {
    cgrad_backend_type type;

    // Pointer to function that allocates a backend-specific tensor handle
    void* (*alloc_tensor_handle)(void);

    // Function pointers for tensor operations (for a specific data type)
    int  (*tensor_init)(void* t, const uint32_t* shape);
    int  (*tensor_fill_rand)(void* t);
    int  (*tensor_gemm)(void* a, void* b, void* c);
    void (*tensor_free)(void* t);
    void (*tensor_print)(const void* t);
    void (*tensor_transpose)(void* t, const uint32_t* perm);
    int  (*tensor_add)(void* a, void* b, void* c);

    int  (*tensor_shallow_copy)(const void* src, void* dst);
    cgrad_tensor_layout* (*tensor_get_layout)(void* t);

    // Add more ops as needed
} cgrad_backend;

#endif // CGRAD_BACKEND_H

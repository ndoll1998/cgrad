#ifndef CGRAD_BACKEND_H
#define CGRAD_BACKEND_H

#include <stdint.h>
#include <stddef.h>

// Enum for backend types
typedef enum {
    CGRAD_BACKEND_F32_CPU = 0,
    // Add more backends here (e.g., CGRAD_BACKEND_CUDA, CGRAD_BACKEND_METAL, ...)
} cgrad_backend_type;

struct cgrad_tensor_layout;
struct cgrad_tensor_f32;

typedef struct cgrad_backend {
    cgrad_backend_type type;

    // Function pointers for tensor operations (for a specific data type)
    int  (*init)(void* t, const uint32_t* shape);
    int  (*fill_rand)(void* t);
    int  (*gemm)(void* a, void* b, void* c);
    void (*free)(void* t);
    void (*print)(const void* t);
    void (*transpose)(void* t, const uint32_t* perm);
    // Add more ops as needed
} cgrad_backend;

#endif // CGRAD_BACKEND_H

#ifndef CGRAD_BACKEND_H
#define CGRAD_BACKEND_H

#include <stdint.h>
#include <stddef.h>
#include "cgrad_layout.h"

/**
 * @brief Enum for backend types.
 */
typedef enum {
    CGRAD_BACKEND_F32_CPU = 0,
} cgrad_backend_type;

/**
 * @brief Backend interface for tensor operations.
 * 
 * Function pointers are grouped by logical operation type for clarity and consistency.
 */
typedef struct cgrad_backend {
    cgrad_backend_type type;

    /**
     * @brief Allocate a backend-specific tensor handle.
     * @return Pointer to the allocated tensor handle.
     */
    void* (*alloc_tensor_handle)(void);

    // --- Initialization/Allocation ---
    /**
     * @brief Initialize a tensor with the given shape.
     */
    int  (*tensor_init)(void* t, const uint32_t* shape);

    // --- Randomization ---
    /**
     * @brief Fill the tensor with random values.
     */
    int  (*tensor_fill_rand)(void* t);

    // --- Memory/Copy ---
    /**
     * @brief Create a shallow copy of a tensor handle (deep copy layout, shallow copy data).
     */
    int  (*tensor_shallow_copy)(const void* src, void* dst);

    /**
     * @brief Free the memory associated with a tensor.
     */
    void (*tensor_free)(void* t);

    // --- Math Ops ---
    /**
     * @brief Add two tensors elementwise and store the result in a third tensor.
     */
    int  (*tensor_add)(void* a, void* b, void* c);

    /**
     * @brief Perform batched matrix multiplication (GEMM) on two tensors.
     */
    int  (*tensor_gemm)(void* a, void* b, void* c);

    // --- Data Access/Info ---
    /**
     * @brief Get the layout of a tensor handle.
     */
    cgrad_tensor_layout* (*tensor_get_layout)(void* t);

    /**
     * @brief Print the tensor's shape and contents.
     */
    void (*tensor_print)(const void* t);

    // --- Transform ---
    /**
     * @brief Transpose the tensor according to the given permutation.
     */
    void (*tensor_transpose)(void* t, const uint32_t* perm);

    // Add more ops as needed (e.g., build_batch_array, ptr, set, contiguous, etc.)
} cgrad_backend;

#endif // CGRAD_BACKEND_H
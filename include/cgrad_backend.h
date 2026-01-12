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
     * @brief Initialize a tensor with the given shape and ndim.
     * @param t Pointer to tensor.
     * @param shape Array of dimensions (length ndim).
     * @param ndim Number of dimensions in shape (≤ MAX_TENSOR_DIM).
     */
    int  (*tensor_init)(void* t, const uint32_t* shape, int ndim);

    /**
     * @brief Fill the tensor with a constant value.
     * @param t Pointer to tensor.
     * @param value The value to fill the tensor with.
     */
    int  (*tensor_fill)(void* t, float value);

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
     * @brief Make a contiguous copy of a tensor.
     */
    int  (*tensor_contiguous)(const void* src, void* dst);

    /**
     * @brief Free the memory associated with a tensor.
     */
    void (*tensor_free)(void* t);

    // --- Math Ops ---
    /**
     * @brief Add two tensors elementwise and store the result in a third tensor.
     * @param alpha Scaling factor for a (c = alpha * a + b).
     * @param a First input tensor.
     * @param b Second input tensor.
     * @param c Output tensor.
     */
    int  (*tensor_add)(float alpha, void* a, void* b, void* c);

    /**
     * @brief Perform batched matrix multiplication (GEMM) on two tensors.
     */
    int  (*tensor_gemm)(void* a, void* b, void* c);

    // --- Data Access/Info ---
    /**
     * @brief Get the value at the given indices.
     * @param t Pointer to tensor.
     * @param indices Array of indices.
     * @param ndim Number of dimensions in indices.
     * @param out_value Pointer to float where the value will be written.
     */
    int  (*tensor_get)(const void* t, const uint32_t* indices, int ndim, float* out_value);

    /**
     * @brief Set the value at the given indices.
     * @param t Pointer to tensor.
     * @param indices Array of indices.
     * @param ndim Number of dimensions in indices.
     * @param value Value to set.
     */
    int  (*tensor_set)(void* t, const uint32_t* indices, int ndim, float value);

    /**
     * @brief Get the layout of a tensor handle.
     */
    cgrad_tensor_layout* (*tensor_get_layout)(void* t);

    /**
     * @brief Print only the tensor's data (no shape).
     */
    void (*tensor_print_data)(const void* t);

    // Add more ops as needed (e.g., build_batch_array, ptr, set, contiguous, etc.)
} cgrad_backend;

#endif // CGRAD_BACKEND_H

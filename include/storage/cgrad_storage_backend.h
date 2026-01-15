#ifndef CGRAD_STORAGE_BACKEND_H
#define CGRAD_STORAGE_BACKEND_H

#include <stdint.h>
#include <stddef.h>
#include "storage/cgrad_storage_layout.h"

/**
 * @brief Enum for backend types.
 */
typedef enum {
    CGRAD_STORAGE_BACKEND_F32_CPU = 0,
} cgrad_storage_backend_type;

/**
 * @brief Backend interface for storage operations.
 * 
 * Function pointers are grouped by logical operation type for clarity and consistency.
 */
typedef struct cgrad_storage_backend {
    cgrad_storage_backend_type type;

    /**
     * @brief Size of the backend-specific storage handle in bytes.
     * Used for allocating memory in cgrad_storage_init.
     */
    size_t storage_handle_size;

    // --- Initialization/Allocation ---
    /**
     * @brief Initialize a storage with the given shape and ndim.
     * @param t Pointer to storage.
     * @param shape Array of dimensions (length ndim).
     * @param ndim Number of dimensions in shape (≤ MAX_TENSOR_DIM).
     */
    int  (*storage_init)(void* t, const uint32_t* shape, int ndim);

    /**
     * @brief Fill the storage with a constant value.
     * @param t Pointer to storage.
     * @param value The value to fill the storage with.
     */
    int  (*storage_fill)(void* t, float value);

    /**
     * @brief Fill the storage with random values.
     */
    int  (*storage_fill_rand)(void* t);

    // --- Memory/Copy ---
    /**
     * @brief Create a shallow copy of a storage handle (deep copy layout, shallow copy data).
     */
    int  (*storage_shallow_copy)(const void* src, void* dst);

    /**
     * @brief Make a contiguous copy of a storage.
     */
    int  (*storage_contiguous)(const void* src, void* dst);

    /**
     * @brief Free the memory associated with a storage.
     */
    void (*storage_free)(void* t);

    // --- Math Ops ---
    /**
     * @brief Add two storages elementwise and store the result in a third storage.
     * @param alpha Scaling factor for a (c = alpha * a + b).
     * @param a First input storage.
     * @param b Second input storage.
     * @param c Output storage.
     */
    int  (*storage_add)(float alpha, void* a, void* b, void* c);

    /**
     * @brief Perform batched matrix multiplication (GEMM) on two storages.
     * @param alpha Scaling factor for the matrix product (c = alpha * a * b + beta * c).
     * @param a First input storage.
     * @param b Second input storage.
     * @param beta Scaling factor for the output storage.
     * @param c Output storage.
     */
    int  (*storage_gemm)(float alpha, void* a, void* b, float beta, void* c);

    // --- Data Access/Info ---
    /**
     * @brief Get the value at the given indices.
     * @param t Pointer to storage.
     * @param indices Array of indices.
     * @param ndim Number of dimensions in indices.
     * @param out_value Pointer to float where the value will be written.
     */
    int  (*storage_get)(const void* t, const uint32_t* indices, int ndim, float* out_value);

    /**
     * @brief Set the value at the given indices.
     * @param t Pointer to storage.
     * @param indices Array of indices.
     * @param ndim Number of dimensions in indices.
     * @param value Value to set.
     */
    int  (*storage_set)(void* t, const uint32_t* indices, int ndim, float value);

    /**
     * @brief Get the layout of a storage handle.
     */
    cgrad_storage_layout* (*storage_get_layout)(void* t);

    /**
     * @brief Print only the storage's data (no shape).
     */
    void (*storage_print_data)(const void* t);

    // Add more ops as needed (e.g., build_batch_array, ptr, set, contiguous, etc.)
} cgrad_storage_backend;

/**
 * @brief Get the backend for a given backend type.
 * @param type Backend type.
 * @return Pointer to the backend.
 */
cgrad_storage_backend* cgrad_get_backend(cgrad_storage_backend_type type);

#endif // CGRAD_STORAGE_BACKEND_H

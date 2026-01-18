#ifndef CGRAD_BACKEND_H
#define CGRAD_BACKEND_H

#include "storage/cgrad_storage_layout.h"
#include "third_party/uthash.h"
#include <stdint.h>
#include <stddef.h>

/**
 * @file cgrad_backend.h
 * @brief Backend interface for storage operations.
 * 
 * This header defines the backend interface that all storage backends must implement.
 * Backends are responsible for managing data storage and performing operations on that data.
 */

/**
 * @brief Backend interface for storage operations.
 * 
 * Function pointers are grouped by logical operation type for clarity and consistency.
 * The struct is designed to be hashable directly using uthash, with the name field
 * serving as the hash key.
 */
typedef struct cgrad_backend {
    const char* name;                /**< Backend name (e.g., "cpu_f32") - hash key */

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
     * @brief Compute y = alpha * x + y (AXPY operation).
     * @param alpha Scaling factor for x.
     * @param x First input storage (read-only).
     * @param y Second input storage (modified in-place).
     */
    int  (*storage_axpy)(float alpha, void* x, void* y);

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

    /**
     * @brief uthash handle for direct hashing of backend structs.
     * This allows backends to be stored directly in the hash table without
     * requiring a separate wrapper struct.
     */
    UT_hash_handle hh;
} cgrad_backend;

#endif // CGRAD_BACKEND_H

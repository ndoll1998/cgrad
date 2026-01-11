#ifndef CGRAD_TENSOR_H
#define CGRAD_TENSOR_H

#include "cgrad_backend.h"
#include <stdint.h>
#include <stddef.h>

/**
 * @brief High-level tensor object supporting multiple backends.
 */
typedef struct cgrad_tensor {
    cgrad_backend* backend; /**< Pointer to backend ops */
    void* handle;           /**< Backend-specific tensor object (e.g., cgrad_tensor_f32*) */
} cgrad_tensor;

// --- Initialization/Allocation ---

/**
 * @brief Initialize a high-level tensor with the given shape, ndim, and backend type.
 *        The user-specified shape (length ndim) is placed at the end; leading unspecified dims are set to 1.
 *        For example, shape={3,4}, ndim=2, MAX_TENSOR_DIM=4 => layout.shape={1,1,3,4}
 * @param t Pointer to tensor to initialize.
 * @param shape Array of dimensions (length ndim).
 * @param ndim Number of dimensions in shape (≤ MAX_TENSOR_DIM).
 * @param backend_type Backend type to use.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_init(cgrad_tensor* t, const uint32_t* shape, int ndim, cgrad_backend_type backend_type);

/**
 * @brief Perform a shallow copy of a tensor (copies handle, not data).
 * @param src Source tensor.
 * @param dst Destination tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_shallow_copy(const cgrad_tensor* src, cgrad_tensor* dst);

/**
 * @brief Free the memory associated with a high-level tensor.
 * @param t Pointer to tensor.
 */
void cgrad_tensor_free(cgrad_tensor* t);

/**
 * @brief Fill the tensor with a constant value.
 * @param t Pointer to tensor.
 * @param value The value to fill the tensor with.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_fill(cgrad_tensor* t, float value);

/**
 * @brief Fill the tensor with random values.
 * @param t Pointer to tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_fill_rand(cgrad_tensor* t);

// --- Math Ops ---

/**
 * @brief Perform batched matrix multiplication (GEMM) on two tensors.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_gemm(const cgrad_tensor* a, const cgrad_tensor* b, cgrad_tensor* r);

/**
 * @brief Add two tensors elementwise and store the result in a third tensor.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */

int cgrad_tensor_add(cgrad_tensor* a, cgrad_tensor* b, cgrad_tensor* r);

/**
 * @brief Subtract two tensors elementwise and store the result in a third tensor.
 *        Computes r = a - b.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_sub(cgrad_tensor* a, cgrad_tensor* b, cgrad_tensor* r);

// --- Data Access/Info ---

/**
 * @brief Print the tensor's shape and contents.
 * @param t Pointer to tensor.
 */
void cgrad_tensor_print(const cgrad_tensor* t);

/**
 * @brief Reshape a tensor, using layout reshape and backend copy ops.
 * @param src Source tensor.
 * @param dst Destination tensor.
 * @param new_shape Array of new dimensions (length ndim, may contain one -1).
 * @param ndim Number of dimensions in new_shape (<= TENSOR_DIM).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_reshape(const cgrad_tensor* src, cgrad_tensor* dst, const int32_t* new_shape, int ndim);

// --- Transform ---

/**
 * @brief Transpose the tensor according to the given permutation, applied to the last ndim dims.
 * @param t Pointer to tensor.
 * @param perm Permutation array (length ndim).
 * @param ndim Number of trailing dimensions to permute (≤ MAX_TENSOR_DIM).
 */
void cgrad_tensor_transpose(cgrad_tensor* t, const uint32_t* perm, int ndim);

// --- Backend Registry (for internal use) ---

/**
 * @brief Get the backend for a given backend type.
 * @param type Backend type.
 * @return Pointer to the backend.
 */
cgrad_backend* cgrad_get_backend(cgrad_backend_type type);

#endif // CGRAD_TENSOR_H

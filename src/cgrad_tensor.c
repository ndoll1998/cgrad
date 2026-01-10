#include "cgrad_tensor.h"
#include "cgrad_errors.h"
#include "cgrad_layout.h"
#include <stdlib.h>
#include <string.h>

/**
 * @brief Initialize a high-level tensor with the given shape and backend type.
 * @param t Pointer to tensor to initialize.
 * @param shape Array of dimensions.
 * @param backend_type Backend type to use.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_init(cgrad_tensor* t, const uint32_t* shape, cgrad_backend_type backend_type) {
    if (!t || !shape) return CGRAD_TENSOR_ERR_NULL_POINTER;
    t->backend = cgrad_get_backend(backend_type);
    if (!t->backend) return CGRAD_TENSOR_ERR_BACKEND_MISMATCH;

    // Use backend's tensor handle allocator
    void* handle = t->backend->alloc_tensor_handle();
    if (!handle) return CGRAD_TENSOR_ERR_HANDLE_UNINITIALIZED;
    if (t->backend->tensor_init(handle, shape)) {
        free(handle);
        return CGRAD_TENSOR_ERR_NOT_IMPLEMENTED;
    }
    t->handle = handle;
    return CGRAD_SUCCESS;
}

/**
 * @brief Free the memory associated with a high-level tensor.
 * @param t Pointer to tensor.
 */
void cgrad_tensor_free(cgrad_tensor* t) {
    if (!t || !t->backend || !t->handle) return;
    t->backend->tensor_free(t->handle);
    free(t->handle);
    t->handle = NULL;
}

/**
 * @brief Fill the tensor with a constant value.
 * @param t Pointer to tensor.
 * @param value The value to fill the tensor with.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_fill(cgrad_tensor* t, float value) {
    if (!t || !t->backend || !t->handle) return CGRAD_TENSOR_ERR_NULL_POINTER;
    if (!t->backend->tensor_fill) return CGRAD_TENSOR_ERR_NOT_IMPLEMENTED;
    return t->backend->tensor_fill(t->handle, value);
}

/**
 * @brief Fill the tensor with random values.
 * @param t Pointer to tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_fill_rand(cgrad_tensor* t) {
    if (!t || !t->backend || !t->handle) return CGRAD_TENSOR_ERR_NULL_POINTER;
    return t->backend->tensor_fill_rand(t->handle);
}

/**
 * @brief Perform batched matrix multiplication (GEMM) on two tensors.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_gemm(
    const cgrad_tensor* a,
    const cgrad_tensor* b,
    cgrad_tensor* r
) {
    // validate tensors
    if (!a || !b || !r) return CGRAD_TENSOR_ERR_NULL_POINTER;
    if (!a->backend || !b->backend) return CGRAD_TENSOR_ERR_NULL_POINTER;
    if (a->backend != b->backend) return CGRAD_TENSOR_ERR_BACKEND_MISMATCH;

    // create shallow copies of a and b
    cgrad_tensor a_bcast = *a;
    cgrad_tensor b_bcast = *b;
    a->backend->tensor_shallow_copy(a->handle, a_bcast.handle);
    b->backend->tensor_shallow_copy(b->handle, b_bcast.handle);

    // broadcast layouts
    int bcast_err = cgrad_tensor_layout_broadcast(
        a_bcast.backend->tensor_get_layout(a_bcast.handle),
        b_bcast.backend->tensor_get_layout(b_bcast.handle),
        0,
        MAX_TENSOR_DIM - 2
    );
    if (bcast_err != CGRAD_SUCCESS) return bcast_err;

    // build output shape
    uint32_t r_shape[MAX_TENSOR_DIM];
    memcpy(r_shape, a_bcast.backend->tensor_get_layout(a_bcast.handle)->shape, sizeof(uint32_t) * (MAX_TENSOR_DIM - 2));
    r_shape[MAX_TENSOR_DIM - 2] = a_bcast.backend->tensor_get_layout(a_bcast.handle)->shape[MAX_TENSOR_DIM - 2]; // m
    r_shape[MAX_TENSOR_DIM - 1] = b_bcast.backend->tensor_get_layout(b_bcast.handle)->shape[MAX_TENSOR_DIM - 1]; // n

    // check if r is initialized, if not initialize it
    if (!r->handle) {
        cgrad_tensor_init(r, r_shape, a->backend->type);
    } else {
        // TODO: support writing to existing tensor with matching shape
        return CGRAD_TENSOR_ERR_NOT_IMPLEMENTED;
    }

    return a->backend->tensor_gemm(a->handle, b->handle, r->handle);
}

/**
 * @brief Add two tensors elementwise and store the result in a third tensor.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_add(
    cgrad_tensor* a,
    cgrad_tensor* b,
    cgrad_tensor* r
) {
    // validate tensors
    if (!a || !b || !r) return CGRAD_TENSOR_ERR_NULL_POINTER;
    if (!a->backend || !b->backend) return CGRAD_TENSOR_ERR_NULL_POINTER;
    if (a->backend != b->backend) return CGRAD_TENSOR_ERR_BACKEND_MISMATCH;

    // create shallow copies of a and b
    cgrad_tensor a_bcast = *a;
    cgrad_tensor b_bcast = *b;
    a->backend->tensor_shallow_copy(a->handle, a_bcast.handle);
    b->backend->tensor_shallow_copy(b->handle, b_bcast.handle);

    // broadcast layouts
    int bcast_err = cgrad_tensor_layout_broadcast(
        a_bcast.backend->tensor_get_layout(a_bcast.handle),
        b_bcast.backend->tensor_get_layout(b_bcast.handle),
        0,
        MAX_TENSOR_DIM
    );
    if (bcast_err != CGRAD_SUCCESS) return bcast_err;

    // check if r is initialized, if not initialize it
    if (!r->handle) {
        const uint32_t* shape = a_bcast.backend->tensor_get_layout(a_bcast.handle)->shape;
        cgrad_tensor_init(r, shape, a->backend->type);
    } else {
        // TODO: support writing to existing tensor with matching shape
        return CGRAD_TENSOR_ERR_NOT_IMPLEMENTED;
    }

    return a->backend->tensor_add(a->handle, b->handle, r->handle);
}

/**
 * @brief Print the tensor's shape and contents.
 * @param t Pointer to tensor.
 */
void cgrad_tensor_print(const cgrad_tensor* t) {
    if (!t || !t->backend || !t->handle) return;
    t->backend->tensor_print(t->handle);
}

/**
 * @brief Transpose the tensor according to the given permutation.
 * @param t Pointer to tensor.
 * @param perm Permutation array.
 */
void cgrad_tensor_transpose(cgrad_tensor* t, const uint32_t* perm) {
    if (!t || !t->backend || !t->handle) return;
    t->backend->tensor_transpose(t->handle, perm);
}
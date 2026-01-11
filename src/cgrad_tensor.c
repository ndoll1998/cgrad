#include "cgrad_tensor.h"
#include "cgrad_errors.h"
#include "cgrad_layout.h"
#include "cgrad_tensor_registry.h"
#include <stdlib.h>
#include <string.h>

/**
 * @brief Initialize a high-level tensor with the given shape and backend type.
 * @param t Pointer to tensor to initialize.
 * @param shape Array of dimensions.
 * @param backend_type Backend type to use.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_init(cgrad_tensor* t, const uint32_t* shape, int ndim, cgrad_backend_type backend_type) {
    if (!t || !shape) return CGRAD_TENSOR_ERR_NULL_POINTER;
    t->backend = cgrad_get_backend(backend_type);
    if (!t->backend) return CGRAD_TENSOR_ERR_BACKEND_MISMATCH;

    // Use backend's tensor handle allocator
    void* data = t->backend->alloc_tensor_handle();
    if (!data) return CGRAD_TENSOR_ERR_HANDLE_UNINITIALIZED;
    
    int err = t->backend->tensor_init(data, shape, ndim);
    if (err != CGRAD_SUCCESS) {
        free(data);
        return err;
    }
    
    t->data = data;
    // Register tensor in global registry as a new root
    cgrad_tensor_registry_register(t, NULL);
    return CGRAD_SUCCESS;
}

/**
 * @brief Perform a shallow copy of a tensor (copies handle, not data).
 * @param dst Destination tensor.
 * @param src Source tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_shallow_copy(const cgrad_tensor* src, cgrad_tensor* dst) {
    if (!dst || !src || !src->backend || !src->data)
        return CGRAD_TENSOR_ERR_NULL_POINTER;
    dst->backend = src->backend;
    if (!dst->data) {
        dst->data = dst->backend->alloc_tensor_handle();
        if (!dst->data)
            return CGRAD_TENSOR_ERR_HANDLE_UNINITIALIZED;
    }
    if (!dst->backend->tensor_shallow_copy)
        return CGRAD_TENSOR_ERR_NOT_IMPLEMENTED;
    int err = dst->backend->tensor_shallow_copy(src->data, dst->data);
    if (err != CGRAD_SUCCESS)
        return err;
    // Register the shallow copy in the registry with src as parent
    return cgrad_tensor_registry_register(dst, src);
}

/**
 * @brief Free the memory associated with a high-level tensor.
 *        Returns the error code from the registry deregistration.
 * @param t Pointer to tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_free(cgrad_tensor* t) {
    if (!t || !t->backend || !t->data) return CGRAD_TENSOR_ERR_NULL_POINTER;
    // unregister from tensor registry
    cgrad_tensor* root = NULL;
    int err = cgrad_tensor_registry_deregister(t, &root);
    if (err != CGRAD_SUCCESS)
        return err;
    // free tensor handle if this is the root tensor
    if (root) {
        t->backend->tensor_free(root->data);
    }
    // free the tensor handle
    free(t->data);
    t->data = NULL;
    return CGRAD_SUCCESS;
}

/**
 * @brief Fill the tensor with a constant value.
 * @param t Pointer to tensor.
 * @param value The value to fill the tensor with.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_fill(cgrad_tensor* t, float value) {
    if (!t || !t->backend || !t->data) return CGRAD_TENSOR_ERR_NULL_POINTER;
    if (!t->backend->tensor_fill) return CGRAD_TENSOR_ERR_NOT_IMPLEMENTED;
    return t->backend->tensor_fill(t->data, value);
}

/**
 * @brief Fill the tensor with random values.
 * @param t Pointer to tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_fill_rand(cgrad_tensor* t) {
    if (!t || !t->backend || !t->data) return CGRAD_TENSOR_ERR_NULL_POINTER;
    return t->backend->tensor_fill_rand(t->data);
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
    a->backend->tensor_shallow_copy(a->data, a_bcast.data);
    b->backend->tensor_shallow_copy(b->data, b_bcast.data);

    // broadcast layouts
    int bcast_err = cgrad_tensor_layout_broadcast(
        a_bcast.backend->tensor_get_layout(a_bcast.data),
        b_bcast.backend->tensor_get_layout(b_bcast.data),
        0,
        TENSOR_DIM - 2
    );
    if (bcast_err != CGRAD_SUCCESS) return bcast_err;

    // build output shape
    uint32_t r_shape[TENSOR_DIM];
    memcpy(r_shape, a_bcast.backend->tensor_get_layout(a_bcast.data)->shape, sizeof(uint32_t) * (TENSOR_DIM - 2));
    r_shape[TENSOR_DIM - 2] = a_bcast.backend->tensor_get_layout(a_bcast.data)->shape[TENSOR_DIM - 2]; // m
    r_shape[TENSOR_DIM - 1] = b_bcast.backend->tensor_get_layout(b_bcast.data)->shape[TENSOR_DIM - 1]; // n

    // check if r is initialized, if not initialize it
    if (!r->data) {
        cgrad_tensor_init(r, r_shape, TENSOR_DIM, a->backend->type);
    } else {
        // TODO: support writing to existing tensor with matching shape
        return CGRAD_TENSOR_ERR_NOT_IMPLEMENTED;
    }

    return a->backend->tensor_gemm(a->data, b->data, r->data);
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
    a->backend->tensor_shallow_copy(a->data, a_bcast.data);
    b->backend->tensor_shallow_copy(b->data, b_bcast.data);

    // broadcast layouts
    int bcast_err = cgrad_tensor_layout_broadcast(
        a_bcast.backend->tensor_get_layout(a_bcast.data),
        b_bcast.backend->tensor_get_layout(b_bcast.data),
        0,
        TENSOR_DIM
    );
    if (bcast_err != CGRAD_SUCCESS) return bcast_err;

    // check if r is initialized, if not initialize it
    if (!r->data) {
        const uint32_t* shape = a_bcast.backend->tensor_get_layout(a_bcast.data)->shape;
        cgrad_tensor_init(r, shape, TENSOR_DIM, a->backend->type);
    } else {
        // TODO: support writing to existing tensor with matching shape
        return CGRAD_TENSOR_ERR_NOT_IMPLEMENTED;
    }

    return a->backend->tensor_add(1.0f, a->data, b->data, r->data);
}

/**
 * @brief Subtract two tensors elementwise and store the result in a third tensor.
 *        Computes r = a - b.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_sub(
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
    a->backend->tensor_shallow_copy(a->data, a_bcast.data);
    b->backend->tensor_shallow_copy(b->data, b_bcast.data);

    // broadcast layouts
    int bcast_err = cgrad_tensor_layout_broadcast(
        a_bcast.backend->tensor_get_layout(a_bcast.data),
        b_bcast.backend->tensor_get_layout(b_bcast.data),
        0,
        TENSOR_DIM
    );
    if (bcast_err != CGRAD_SUCCESS) return bcast_err;

    // check if r is initialized, if not initialize it
    if (!r->data) {
        const uint32_t* shape = a_bcast.backend->tensor_get_layout(a_bcast.data)->shape;
        cgrad_tensor_init(r, shape, TENSOR_DIM, a->backend->type);
    } else {
        // TODO: support writing to existing tensor with matching shape
        return CGRAD_TENSOR_ERR_NOT_IMPLEMENTED;
    }

    // r = a - b = a + (-1.0) * b
    return a->backend->tensor_add(-1.0f, b->data, a->data, r->data);
}

/**
 * @brief Print the tensor's shape and contents.
 * @param t Pointer to tensor.
 */
void cgrad_tensor_print(const cgrad_tensor* t) {
    if (!t || !t->backend || !t->data) return;
    t->backend->tensor_print(t->data);
}

/**
 * @brief Reshape a tensor, using layout reshape and backend copy ops.
 * @param src Source tensor.
 * @param dst Destination tensor.
 * @param new_shape Array of new dimensions (length ndim, may contain one -1).
 * @param ndim Number of dimensions in new_shape (<= TENSOR_DIM).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_reshape(const cgrad_tensor* src, cgrad_tensor* dst, const int32_t* new_shape, int ndim) {
    if (!src || !src->backend || !src->data)
        return CGRAD_TENSOR_ERR_NULL_POINTER;

    if (dst->data) {
        // not supported yet
        return CGRAD_TENSOR_ERR_NOT_IMPLEMENTED;
    }

    if (cgrad_tensor_layout_is_regular(src->backend->tensor_get_layout(src->data))) {
        // If src is regular, we can do a shallow copy
        if (!src->backend->tensor_shallow_copy)
            return CGRAD_TENSOR_ERR_NOT_IMPLEMENTED;

        int err = cgrad_tensor_shallow_copy(src, dst);
        if (err != CGRAD_SUCCESS)
            return err;
    } else {
        // If src is not regular, make a contiguous copy into dst
        if (!src->backend->tensor_contiguous)
            return CGRAD_TENSOR_ERR_NOT_IMPLEMENTED;

        uint32_t* src_shape = src->backend->tensor_get_layout(src->data)->shape;
        int err = cgrad_tensor_init(dst, src_shape, TENSOR_DIM, src->backend->type);
        if (err != CGRAD_SUCCESS)
            return err;
        
        err = src->backend->tensor_contiguous(src->data, dst->data);
        if (err != CGRAD_SUCCESS)
            return err;
    }

    return cgrad_tensor_layout_reshape(
        src->backend->tensor_get_layout(dst->data),
        new_shape,
        ndim
    );
}

/**
 * @brief Transpose the tensor according to the given permutation, applied to the last ndim dims.
 * @param t Pointer to tensor.
 * @param perm Permutation array (length ndim).
 * @param ndim Number of trailing dimensions to permute (≤ TENSOR_DIM).
 */
void cgrad_tensor_transpose(cgrad_tensor* t, const uint32_t* perm, int ndim) {
    if (!t || !t->backend || !t->data) return;
    cgrad_tensor_layout_transpose(t->backend->tensor_get_layout(t->data), perm, ndim);
}

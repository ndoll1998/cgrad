#include "storage/cgrad_storage.h"
#include "cgrad_errors.h"
#include "storage/cgrad_storage_layout.h"
#include "storage/cgrad_storage_registry.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

// ============================================================================
// Global Storage Registry
// ============================================================================

static cgrad_storage_registry* g_global_registry = NULL;

/**
 * @brief Get or create the global storage registry.
 * This is exposed for testing purposes.
 */
cgrad_storage_registry* get_global_registry(void) {
    if (g_global_registry == NULL) {
        g_global_registry = (cgrad_storage_registry*)malloc(sizeof(cgrad_storage_registry));
        if (g_global_registry == NULL) {
            return NULL;
        }
        
        int err = cgrad_storage_registry_init(g_global_registry);
        if (err != CGRAD_SUCCESS) {
            free(g_global_registry);
            g_global_registry = NULL;
            return NULL;
        }
    }
    
    return g_global_registry;
}

/**
 * @brief Cleanup the global storage registry.
 */
void cgrad_storage_cleanup_global_registry(void) {
    if (g_global_registry != NULL) {
        cgrad_storage_registry_free(g_global_registry);
        free(g_global_registry);
        g_global_registry = NULL;
    }
}

// ============================================================================
// Storage Recording Wrappers
// ============================================================================

/**
 * @brief Start recording storage allocations.
 */
cgrad_storage_registry_record* cgrad_storage_start_recording(void) {
    cgrad_storage_registry* registry = get_global_registry();
    if (!registry) return NULL;
    
    return cgrad_storage_registry_start_recording(registry);
}

/**
 * @brief Stop recording storage allocations.
 */
int cgrad_storage_stop_recording(cgrad_storage_registry_record* record) {
    if (!record) return CGRAD_ERR_NULL_POINTER;
    
    cgrad_storage_registry* registry = get_global_registry();
    if (!registry) return CGRAD_ERR_NULL_POINTER;
    
    // Stop recording
    return cgrad_storage_registry_stop_recording(registry, record);
}

/**
 * @brief Free all storages recorded in a record.
 */
int cgrad_storage_free_all_from_record(cgrad_storage_registry_record* record) {
    if (!record) return CGRAD_ERR_NULL_POINTER;

    int first_error = CGRAD_SUCCESS;
    cgrad_storage_registry_node *entry, *tmp;
    HASH_ITER(hh, record->storage_map, entry, tmp) {
        // Only free the record entry struct, not the underlying storage
        HASH_DEL(record->storage_map, entry);
        free(entry);
    }

    return first_error;
}

/**
 * @brief Initialize a high-level tensor with the given shape and backend type.
 * @param t Pointer to tensor to initialize.
 * @param shape Array of dimensions.
 * @param backend_type Backend type to use.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_init(cgrad_storage* t, const uint32_t* shape, int ndim, const char* backend_name) {
    if (!t || !shape) return CGRAD_ERR_NULL_POINTER;

    // Get backend
    cgrad_storage_backend* backend = cgrad_get_backend(backend_name);
    if (!backend) return CGRAD_STORAGE_ERR_BACKEND_MISMATCH;

    // Allocate tensor handle using the backend's handle size
    void* data = calloc(1, backend->storage_handle_size);
    if (!data) return CGRAD_STORAGE_ERR_HANDLE_UNINITIALIZED;
    
    // initialize the tensor
    int err = backend->storage_init(data, shape, ndim);
    if (err != CGRAD_SUCCESS) {
        free(data);
        return err;
    }
    
    // populate tensor attributes
    uuid_generate(t->uuid);
    t->data = data;
    t->backend = backend;

    // Register tensor in global registry as a new root
    cgrad_storage_registry* registry = get_global_registry();
    if (registry) {
        cgrad_storage_registry_register(registry, t, NULL);
    }
    return CGRAD_SUCCESS;
}

/**
 * @brief Perform a shallow copy of a tensor (copies handle, not data).
 * @param dst Destination tensor.
 * @param src Source tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_shallow_copy(const cgrad_storage* src, cgrad_storage* dst) {
    if (!dst || !src || !src->backend || !src->data) {
        return CGRAD_ERR_NULL_POINTER;
    }
    if (!src->backend->storage_shallow_copy)
        return CGRAD_ERR_NOT_IMPLEMENTED;
    
    // Allocate tensor handle using the backend's handle size
    void* data = calloc(1, src->backend->storage_handle_size);
    if (!data)
        return CGRAD_STORAGE_ERR_HANDLE_UNINITIALIZED;
    
    int err = src->backend->storage_shallow_copy(src->data, data);
    if (err != CGRAD_SUCCESS) {
        free(data);
        return err;
    }

    // populate tensor attributes
    uuid_generate(dst->uuid);
    dst->data = data;
    dst->backend = src->backend;

    // Register the shallow copy in the registry with src as parent
    cgrad_storage_registry* registry = get_global_registry();
    if (registry) {
        return cgrad_storage_registry_register(registry, dst, src);
    }
    return CGRAD_SUCCESS;
}

/**
 * @brief Free the memory associated with a high-level tensor.
 *        Returns the error code from the registry deregistration.
 * @param t Pointer to tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_free(cgrad_storage* t) {
    if (!t || !t->backend || !t->data) return CGRAD_ERR_NULL_POINTER;

    cgrad_storage_registry* registry = get_global_registry();
    if (!registry) return CGRAD_ERR_NULL_POINTER;

    // get the root tensor (holding the data)
    cgrad_storage root;
    int err = cgrad_storage_registry_get_root(registry, t, &root);
    if (err != CGRAD_SUCCESS) return err;

    if (cgrad_storage_registry_bucket_get_size(registry, t) == 1) {
        // this is the only tensor in the bucket
        // free the root
        t->backend->storage_free(root.data);
        // delete the whole bucket
        err = cgrad_storage_registry_deregister_and_delete_bucket(registry, t);
        if (err != CGRAD_SUCCESS) return err;
        // free root handle after delete
        free(t->data);
        t->data = NULL;
    } else {
        // deregister the tensor
        err = cgrad_storage_registry_deregister(registry, t);
        if (err != CGRAD_SUCCESS) return err;
        // free the tensor handle if this is not the root tensor
        if (uuid_compare(t->uuid, root.uuid)) {
            free(t->data);
            t->data = NULL;
        }
    }
    
    return CGRAD_SUCCESS;
}

/**
 * @brief Fill the tensor with a constant value.
 * @param t Pointer to tensor.
 * @param value The value to fill the tensor with.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_fill(cgrad_storage* t, float value) {
    if (!t || !t->backend || !t->data) return CGRAD_ERR_NULL_POINTER;
    if (!t->backend->storage_fill) return CGRAD_ERR_NOT_IMPLEMENTED;
    return t->backend->storage_fill(t->data, value);
}

/**
 * @brief Fill the tensor with random values.
 * @param t Pointer to tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_fill_rand(cgrad_storage* t) {
    if (!t || !t->backend || !t->data) return CGRAD_ERR_NULL_POINTER;
    return t->backend->storage_fill_rand(t->data);
}

/**
 * @brief Perform batched matrix multiplication (GEMM) on two tensors.
 *        Computes r = alpha * a * b + beta * r.
 * @param alpha Scaling factor for the matrix product.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param beta Scaling factor for the output tensor.
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_gemm(
    float alpha,
    const cgrad_storage* a,
    const cgrad_storage* b,
    float beta,
    cgrad_storage* r
) {
    // validate tensors
    if (!a || !b || !r) return CGRAD_ERR_NULL_POINTER;
    if (!a->backend || !b->backend) return CGRAD_ERR_NULL_POINTER;
    if (a->backend != b->backend) return CGRAD_STORAGE_ERR_BACKEND_MISMATCH;

    // record all storages created here
    cgrad_storage_registry_record* storage_record = cgrad_storage_start_recording();

    // create shallow copies of a and b
    cgrad_storage a_bcast;
    int err = cgrad_storage_shallow_copy(a, &a_bcast);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }
    
    cgrad_storage b_bcast;
    err = cgrad_storage_shallow_copy(b, &b_bcast);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }

    // broadcast layouts
    err = cgrad_storage_layout_broadcast(
        a_bcast.backend->storage_get_layout(a_bcast.data),
        b_bcast.backend->storage_get_layout(b_bcast.data),
        0,
        TENSOR_DIM - 2
    );
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }

    // build output shape
    uint32_t r_shape[TENSOR_DIM];
    memcpy(r_shape, a_bcast.backend->storage_get_layout(a_bcast.data)->shape, sizeof(uint32_t) * (TENSOR_DIM - 2));
    r_shape[TENSOR_DIM - 2] = a_bcast.backend->storage_get_layout(a_bcast.data)->shape[TENSOR_DIM - 2]; // m
    r_shape[TENSOR_DIM - 1] = b_bcast.backend->storage_get_layout(b_bcast.data)->shape[TENSOR_DIM - 1]; // n

    // check if r is initialized, if not initialize it
    if (!r->data) {
        err = cgrad_storage_init(r, r_shape, TENSOR_DIM, a->backend->name);
        if (err != CGRAD_SUCCESS) {
            cgrad_storage_stop_recording(storage_record);
            cgrad_storage_free_all_from_record(storage_record);
            return err;
        }
    } else {
        // Check if result tensor shape matches expected shape
        const cgrad_storage_layout* r_layout = r->backend->storage_get_layout(r->data);
        int shape_matches = 1;
        for (int i = 0; i < TENSOR_DIM; ++i) {
            if (r_layout->shape[i] != r_shape[i]) {
                shape_matches = 0;
                break;
            }
        }
        
        if (!shape_matches) {
            cgrad_storage_stop_recording(storage_record);
            cgrad_storage_free_all_from_record(storage_record);
            return CGRAD_STORAGE_ERR_SHAPE_MISMATCH;
        }
        
        // Check if tensor is contiguous
        if (!cgrad_storage_layout_is_contiguous(r_layout)) {
            cgrad_storage_stop_recording(storage_record);
            cgrad_storage_free_all_from_record(storage_record);
            return CGRAD_ERR_NOT_IMPLEMENTED;
        }
    }

    err = a->backend->storage_gemm(alpha, a->data, b->data, beta, r->data);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }

    // cleanup and do not free output storage
    cgrad_storage_registry_record_remove(storage_record, r);
    cgrad_storage_stop_recording(storage_record);
    return cgrad_storage_free_all_from_record(storage_record);
}

/**
 * @brief Compute y = alpha * x + y (AXPY operation).
 *        Computes r = alpha * x + r.
 * @param alpha Scaling factor for x.
 * @param x First input tensor.
 * @param y Second input tensor (used to initialize r if r is uninitialized).
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_axpy(
    float alpha,
    cgrad_storage* x,
    cgrad_storage* y,
    cgrad_storage* r
) {
    // validate tensors
    if (!x || !y || !r) return CGRAD_ERR_NULL_POINTER;
    if (!x->backend || !y->backend) return CGRAD_ERR_NULL_POINTER;
    if (x->backend != y->backend) return CGRAD_STORAGE_ERR_BACKEND_MISMATCH;
    
    // record all storages created here
    cgrad_storage_registry_record* storage_record = cgrad_storage_start_recording();

    // create shallow copies of x and y
    cgrad_storage x_bcast;
    int err = cgrad_storage_shallow_copy(x, &x_bcast);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }
    cgrad_storage y_bcast;
    err = cgrad_storage_shallow_copy(y, &y_bcast);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }

    // broadcast layouts
    err = cgrad_storage_layout_broadcast(
        x_bcast.backend->storage_get_layout(x_bcast.data),
        y_bcast.backend->storage_get_layout(y_bcast.data),
        0,
        TENSOR_DIM
    );
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }

    // check if r is initialized, if not initialize it
    if (!r->data) {
        const uint32_t* shape = x_bcast.backend->storage_get_layout(x_bcast.data)->shape;
        err = cgrad_storage_init(r, shape, TENSOR_DIM, x->backend->name);
        if (err != CGRAD_SUCCESS) {
            cgrad_storage_stop_recording(storage_record);
            cgrad_storage_free_all_from_record(storage_record);
            return err;
        }
    } else {
        // Check if result tensor shape matches expected shape
        const cgrad_storage_layout* r_layout = r->backend->storage_get_layout(r->data);
        int shape_matches = 1;
        for (int i = 0; i < TENSOR_DIM; ++i) {
            if (r_layout->shape[i] != x_bcast.backend->storage_get_layout(x_bcast.data)->shape[i]) {
                shape_matches = 0;
                break;
            }
        }
        
        if (!shape_matches) {
            cgrad_storage_stop_recording(storage_record);
            cgrad_storage_free_all_from_record(storage_record);
            return CGRAD_STORAGE_ERR_SHAPE_MISMATCH;
        }
        
        // Check if r is contiguous
        // TODO: we can use is_regular instead of contiguous once we implemented
        //       storage_copy as a less strict version of storage_contiguous
        if (!cgrad_storage_layout_is_contiguous(r_layout)) {
            cgrad_storage_stop_recording(storage_record);
            cgrad_storage_free_all_from_record(storage_record);
            return CGRAD_ERR_NOT_IMPLEMENTED;
        }
    }
    
    if (uuid_compare(y_bcast.uuid, r->uuid) != 0) {
        // y and r are different tensors, copy y to r
        err = y_bcast.backend->storage_contiguous(y_bcast.data, r->data);
        if (err != CGRAD_SUCCESS) {
            cgrad_storage_stop_recording(storage_record);
            cgrad_storage_free_all_from_record(storage_record);
            return err;
        }
    }
    
    err = x->backend->storage_axpy(alpha, x->data, r->data, NULL);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }

    // cleanup and do not free output storage
    cgrad_storage_registry_record_remove(storage_record, r);
    cgrad_storage_stop_recording(storage_record);
    return cgrad_storage_free_all_from_record(storage_record);
}


/**
 * @brief Get the value at the given indices.
 * @param t Pointer to storage.
 * @param indices Array of indices.
 * @param ndim Number of dimensions in indices.
 * @param out_value Pointer to float where the value will be written.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_get(const cgrad_storage* t, const uint32_t* indices, int ndim, float* out_value) {
    if (!t || !t->backend || !t->data) return CGRAD_ERR_NULL_POINTER;
    if (!t->backend->storage_get) return CGRAD_ERR_NOT_IMPLEMENTED;
    return t->backend->storage_get(t->data, indices, ndim, out_value);
}

/**
 * @brief Print the tensor's shape and contents.
 * @param t Pointer to tensor.
 */
void cgrad_storage_print(const cgrad_storage* t) {
    if (!t || !t->backend || !t->data) return;
    const cgrad_storage_layout* layout = t->backend->storage_get_layout(t->data);
    printf("Shape: ");
    cgrad_storage_layout_print_shape(layout, TENSOR_DIM);
    t->backend->storage_print_data(t->data);
}

/**
 * @brief Make a contiguous copy of a tensor into dst.
 * @param src Source tensor.
 * @param dst Destination tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_contiguous(const cgrad_storage* src, cgrad_storage* dst) {
    if (!dst || !src || !src->backend || !src->data)
        return CGRAD_ERR_NULL_POINTER;
    if (!src->backend->storage_contiguous)
        return CGRAD_ERR_NOT_IMPLEMENTED;

    if (cgrad_storage_layout_is_contiguous(
        src->backend->storage_get_layout(src->data)
    )) {
        // already contiguous, do a shallow copy
        return cgrad_storage_shallow_copy(src, dst);
    }
    
    // record all storages created here
    cgrad_storage_registry_record* storage_record = cgrad_storage_start_recording();

    uint32_t* src_shape = src->backend->storage_get_layout(src->data)->shape;
    int err = cgrad_storage_init(dst, src_shape, TENSOR_DIM, src->backend->name);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }

    err = src->backend->storage_contiguous(src->data, dst->data);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }

    return CGRAD_SUCCESS;
}

/**
 * @brief Reshape a tensor, using layout reshape and backend copy ops.
 * @param src Source tensor.
 * @param dst Destination tensor.
 * @param new_shape Array of new dimensions (length ndim, may contain one -1).
 * @param ndim Number of dimensions in new_shape (<= TENSOR_DIM).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_reshape(const cgrad_storage* src, cgrad_storage* dst, const int32_t* new_shape, int ndim) {
    if (!src || !src->backend || !src->data)
        return CGRAD_ERR_NULL_POINTER;

    if (dst->data) {
        // TODO: write to existing buffer not supported yet
        return CGRAD_ERR_NOT_IMPLEMENTED;
    }
    
    // record all storages created here
    cgrad_storage_registry_record* storage_record = cgrad_storage_start_recording();

    if (cgrad_storage_layout_is_regular(src->backend->storage_get_layout(src->data))) {
        // If src is regular, we can do a shallow copy
        if (!src->backend->storage_shallow_copy)
            return CGRAD_ERR_NOT_IMPLEMENTED;

        int err = cgrad_storage_shallow_copy(src, dst);
        if (err != CGRAD_SUCCESS) {
            cgrad_storage_stop_recording(storage_record);
            cgrad_storage_free_all_from_record(storage_record);
            return err;
        }
    } else {
        // If src is not regular, make a contiguous copy into dst
        int err = cgrad_storage_contiguous(src, dst);
        if (err != CGRAD_SUCCESS) {
            cgrad_storage_stop_recording(storage_record);
            cgrad_storage_free_all_from_record(storage_record);
            return err;
        }
    }

    return cgrad_storage_layout_reshape(
        dst->backend->storage_get_layout(dst->data),
        new_shape,
        ndim
    );
}

/**
 * @brief Transpose the tensor according to the given permutation, applied to the last ndim dims.
 * Creates a shallow copy of the source tensor and applies the transpose to the layout.
 * @param src Source tensor.
 * @param dst Destination tensor (will be initialized with shallow copy + transpose).
 * @param perm Permutation array (length ndim).
 * @param ndim Number of trailing dimensions to permute (≤ TENSOR_DIM).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_transpose(const cgrad_storage* src, cgrad_storage* dst, const uint32_t* perm, int ndim) {
    if (!src || !dst) return CGRAD_ERR_NULL_POINTER;
    if (!src->backend || !src->data) return CGRAD_ERR_NULL_POINTER;
    
    // track all storages created here
    cgrad_storage_registry_record* storage_record = cgrad_storage_start_recording();

    // Create shallow copy of source
    int err = cgrad_storage_shallow_copy(src, dst);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }
    
    // Apply transpose to the layout
    err = cgrad_storage_layout_transpose(dst->backend->storage_get_layout(dst->data), perm, ndim);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }

    return CGRAD_SUCCESS;
}

/**
 * @brief Sum a tensor over specified axes using reshape and GEMM with a tensor of all ones.
 * @param a Input tensor.
 * @param mask Right-aligned mask (length ndim) indicating which axes to sum (1=sum, 0=keep).
 * @param ndim Number of dimensions in mask (≤ TENSOR_DIM).
 * @param r Output tensor (initialized inside function).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_sum(
    const cgrad_storage* a,
    const uint8_t* mask,
    int ndim,
    cgrad_storage* r
) {
    if (!a || !mask || !r) return CGRAD_ERR_NULL_POINTER;
    if (!a->backend) return CGRAD_ERR_NULL_POINTER;

    // track all storages created here
    cgrad_storage_registry_record* storage_record = cgrad_storage_start_recording();
    
    // get the layout of storage a
    const cgrad_storage_layout* layout = a->backend->storage_get_layout(a->data);
    
    // Compute the target shape using layout reduce
    cgrad_storage_layout target_layout = *layout;
    int err = cgrad_storage_layout_reduce(&target_layout, mask, ndim);
    if (err != CGRAD_SUCCESS) return err;
    
    // Convert to int32_t for reshape
    int32_t target_shape[TENSOR_DIM];
    for (int i = 0; i < TENSOR_DIM; ++i) {
        target_shape[i] = (int32_t)target_layout.shape[i];
    }
    
    // Create full mask for checking which dims are summed
    uint8_t full_mask[TENSOR_DIM] = {0};
    int mask_offset = TENSOR_DIM - ndim;
    for (int i = 0; i < ndim; ++i) {
        full_mask[mask_offset + i] = mask[i];
    }

    // Check if summed dims are already last
    int already_last = 1;
    for (int i = 0; i < TENSOR_DIM - 1; ++i) {
        if (full_mask[i] && !full_mask[i + 1]) {
            already_last = 0;
            break;
        }
    }

    cgrad_storage a_perm = *a;
    cgrad_storage a_transposed = {0};

    if (!already_last) {
        // Compute permutation to move summed dims to the end
        // kept dims first, then summed dims, preserving order
        uint32_t perm[TENSOR_DIM];
        int kept_count = 0, sum_count = 0;
        for (int i = 0; i < TENSOR_DIM; ++i) {
            if (!full_mask[i]) perm[kept_count++] = i;
        }
        for (int i = 0; i < TENSOR_DIM; ++i) {
            if (full_mask[i]) perm[kept_count + sum_count++] = i;
        }
        
        err = cgrad_storage_transpose(&a_perm, &a_transposed, perm, TENSOR_DIM);
        if (err != CGRAD_SUCCESS) {
            cgrad_storage_stop_recording(storage_record);
            cgrad_storage_free_all_from_record(storage_record);
            return err;
        }
        a_perm = a_transposed;
    }

    // Count kept and summed dims
    int kept_count = 0;
    for (int i = 0; i < TENSOR_DIM; ++i) {
        if (!full_mask[i]) kept_count++;
    }

    // Compute dimensions
    int32_t kept_size = 1, summed_size = 1;
    for (int i = 0; i < TENSOR_DIM; ++i) {
        if (full_mask[i]) summed_size *= layout->shape[i];
        else kept_size *= layout->shape[i];
    }

    // Reshape to collapse kept and summed dims
    cgrad_storage a_reshaped = {0};
    err = cgrad_storage_reshape(&a_perm, &a_reshaped, (const int32_t[]){kept_size, summed_size}, 2);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }
    
    // Create ones tensor of shape (summed_size, 1)
    cgrad_storage ones = {0};
    err = cgrad_storage_init(&ones, (const uint32_t[]){summed_size, 1}, 2, a_perm.backend->name);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }
    err = cgrad_storage_fill(&ones, 1.0f);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }
    
    // Compute sum via GEMM
    cgrad_storage r_mat = {0};
    err = cgrad_storage_gemm(1.0f, &a_reshaped, &ones, 0.0f, &r_mat);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }

    // Reshape r_mat back to target shape
    err = cgrad_storage_reshape(&r_mat, r, target_shape, TENSOR_DIM);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_all_from_record(storage_record);
        return err;
    }

    // cleanup and do not free output storage
    cgrad_storage_registry_record_remove(storage_record, r);
    cgrad_storage_stop_recording(storage_record);
    return cgrad_storage_free_all_from_record(storage_record);
}

#include "cgrad_storage.h"
#include "cgrad_errors.h"
#include "cgrad_storage_layout.h"
#include "cgrad_storage_registry.h"
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
// Storage Tracking Wrappers
// ============================================================================

/**
 * @brief Start tracking storage allocations.
 */
cgrad_storage_registry_tracker* cgrad_storage_start_tracking(void) {
    cgrad_storage_registry* registry = get_global_registry();
    if (!registry) return NULL;
    
    return cgrad_storage_registry_start_tracking(registry);
}

/**
 * @brief Stop tracking and free all tracked storages.
 */
int cgrad_storage_stop_tracking_and_free_all(cgrad_storage_registry_tracker* tracker) {
    if (!tracker) return CGRAD_ERR_NULL_POINTER;
    
    cgrad_storage_registry* registry = get_global_registry();
    if (!registry) return CGRAD_ERR_NULL_POINTER;
    
    // Stop tracking first
    int err = cgrad_storage_registry_stop_tracking(registry, tracker);
    if (err != CGRAD_SUCCESS) {
        return err;
    }
    
    // Free all tracked storages
    int first_error = CGRAD_SUCCESS;
    cgrad_storage_entry *entry, *tmp;
    HASH_ITER(hh, tracker->storage_map, entry, tmp) {
        if (entry->storage) {
            int err = cgrad_storage_free(entry->storage);
            if (err != CGRAD_SUCCESS && first_error == CGRAD_SUCCESS) {
                first_error = err; // Remember first error
            }
        }
    }
    
    // Free the tracker itself
    cgrad_storage_registry_tracker_free(tracker);
    
    return first_error;
}

/**
 * @brief Initialize a high-level tensor with the given shape and backend type.
 * @param t Pointer to tensor to initialize.
 * @param shape Array of dimensions.
 * @param backend_type Backend type to use.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_init(cgrad_storage* t, const uint32_t* shape, int ndim, cgrad_storage_backend_type backend_type) {
    if (!t || !shape) return CGRAD_ERR_NULL_POINTER;

    // get the backend
    cgrad_storage_backend* backend = cgrad_get_backend(backend_type);
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

    if (cgrad_storage_registry_get_bucket_size(registry, t) == 1) {
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
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_gemm(
    const cgrad_storage* a,
    const cgrad_storage* b,
    cgrad_storage* r
) {
    // validate tensors
    if (!a || !b || !r) return CGRAD_ERR_NULL_POINTER;
    if (!a->backend || !b->backend) return CGRAD_ERR_NULL_POINTER;
    if (a->backend != b->backend) return CGRAD_STORAGE_ERR_BACKEND_MISMATCH;

    // track all storages created here
    cgrad_storage_registry_tracker* storage_tracker = cgrad_storage_start_tracking();

    // create shallow copies of a and b
    cgrad_storage a_bcast;
    int err = cgrad_storage_shallow_copy(a, &a_bcast);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }
    
    cgrad_storage b_bcast;
    err = cgrad_storage_shallow_copy(b, &b_bcast);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
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
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }

    // build output shape
    uint32_t r_shape[TENSOR_DIM];
    memcpy(r_shape, a_bcast.backend->storage_get_layout(a_bcast.data)->shape, sizeof(uint32_t) * (TENSOR_DIM - 2));
    r_shape[TENSOR_DIM - 2] = a_bcast.backend->storage_get_layout(a_bcast.data)->shape[TENSOR_DIM - 2]; // m
    r_shape[TENSOR_DIM - 1] = b_bcast.backend->storage_get_layout(b_bcast.data)->shape[TENSOR_DIM - 1]; // n

    // check if r is initialized, if not initialize it
    if (!r->data) {
        err = cgrad_storage_init(r, r_shape, TENSOR_DIM, a->backend->type);
        if (err != CGRAD_SUCCESS) {
            cgrad_storage_stop_tracking_and_free_all(storage_tracker);
            return err;
        }
    } else {
        // TODO: Allow writing to existing tensor if shape matches and r is contiguous
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return CGRAD_ERR_NOT_IMPLEMENTED;
    }

    err = a->backend->storage_gemm(a->data, b->data, r->data);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }

    // cleanup and do not free output storage
    cgrad_storage_registry_tracker_remove(storage_tracker, r);
    return cgrad_storage_stop_tracking_and_free_all(storage_tracker);
}

/**
 * @brief Add two tensors elementwise and store the result in a third tensor.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_add(
    cgrad_storage* a,
    cgrad_storage* b,
    cgrad_storage* r
) {
    // validate tensors
    if (!a || !b || !r) return CGRAD_ERR_NULL_POINTER;
    if (!a->backend || !b->backend) return CGRAD_ERR_NULL_POINTER;
    if (a->backend != b->backend) return CGRAD_STORAGE_ERR_BACKEND_MISMATCH;
    
    // track all storages created here
    cgrad_storage_registry_tracker* storage_tracker = cgrad_storage_start_tracking();

    // create shallow copies of a and b
    cgrad_storage a_bcast;
    int err = cgrad_storage_shallow_copy(a, &a_bcast);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }
    cgrad_storage b_bcast;
    err = cgrad_storage_shallow_copy(b, &b_bcast);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }

    // broadcast layouts
    err = cgrad_storage_layout_broadcast(
        a_bcast.backend->storage_get_layout(a_bcast.data),
        b_bcast.backend->storage_get_layout(b_bcast.data),
        0,
        TENSOR_DIM
    );
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }

    // check if r is initialized, if not initialize it
    if (!r->data) {
        const uint32_t* shape = a_bcast.backend->storage_get_layout(a_bcast.data)->shape;
        err = cgrad_storage_init(r, shape, TENSOR_DIM, a->backend->type);
        if (err != CGRAD_SUCCESS) {
            cgrad_storage_stop_tracking_and_free_all(storage_tracker);
            return err;
        }
    } else {
        // TODO: support writing to existing tensor with matching shape
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return CGRAD_ERR_NOT_IMPLEMENTED;
    }
    
    err = a->backend->storage_add(1.0f, a->data, b->data, r->data);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }

    // cleanup and do not free output storage
    cgrad_storage_registry_tracker_remove(storage_tracker, r);
    return cgrad_storage_stop_tracking_and_free_all(storage_tracker);
}

/**
 * @brief Subtract two tensors elementwise and store the result in a third tensor.
 *        Computes r = a - b.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_sub(
    cgrad_storage* a,
    cgrad_storage* b,
    cgrad_storage* r
) {
    // validate tensors
    if (!a || !b || !r) return CGRAD_ERR_NULL_POINTER;
    if (!a->backend || !b->backend) return CGRAD_ERR_NULL_POINTER;
    if (a->backend != b->backend) return CGRAD_STORAGE_ERR_BACKEND_MISMATCH;
    
    // track all storages created here
    cgrad_storage_registry_tracker* storage_tracker = cgrad_storage_start_tracking();

    // create shallow copies of a and b
    cgrad_storage a_bcast;
    int err = cgrad_storage_shallow_copy(a, &a_bcast);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }
    cgrad_storage b_bcast;
    err = cgrad_storage_shallow_copy(b, &b_bcast);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }

    // broadcast layouts
    err = cgrad_storage_layout_broadcast(
        a_bcast.backend->storage_get_layout(a_bcast.data),
        b_bcast.backend->storage_get_layout(b_bcast.data),
        0,
        TENSOR_DIM
    );
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }

    // check if r is initialized, if not initialize it
    if (!r->data) {
        const uint32_t* shape = a_bcast.backend->storage_get_layout(a_bcast.data)->shape;
        err = cgrad_storage_init(r, shape, TENSOR_DIM, a->backend->type);
        if (err != CGRAD_SUCCESS) {
            cgrad_storage_stop_tracking_and_free_all(storage_tracker);
            return err;
        }
    } else {
        // TODO: support writing to existing tensor with matching shape
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return CGRAD_ERR_NOT_IMPLEMENTED;
    }
    
    err = a->backend->storage_add(-1.0f, a->data, b->data, r->data);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }

    // cleanup and do not free output storage
    cgrad_storage_registry_tracker_remove(storage_tracker, r);
    return cgrad_storage_stop_tracking_and_free_all(storage_tracker);
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

    uint32_t* src_shape = src->backend->storage_get_layout(src->data)->shape;
    int err = cgrad_storage_init(dst, src_shape, TENSOR_DIM, src->backend->type);
    if (err != CGRAD_SUCCESS)
        return err;

    err = src->backend->storage_contiguous(src->data, dst->data);
    if (err != CGRAD_SUCCESS)
        return err;

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
    
    // track all storages created here
    cgrad_storage_registry_tracker* storage_tracker = cgrad_storage_start_tracking();

    if (cgrad_storage_layout_is_regular(src->backend->storage_get_layout(src->data))) {
        // If src is regular, we can do a shallow copy
        if (!src->backend->storage_shallow_copy)
            return CGRAD_ERR_NOT_IMPLEMENTED;

        int err = cgrad_storage_shallow_copy(src, dst);
        if (err != CGRAD_SUCCESS) {
            cgrad_storage_stop_tracking_and_free_all(storage_tracker);
            return err;
        }
    } else {
        // If src is not regular, make a contiguous copy into dst
        int err = cgrad_storage_contiguous(src, dst);
        if (err != CGRAD_SUCCESS) {
            cgrad_storage_stop_tracking_and_free_all(storage_tracker);
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
    cgrad_storage_registry_tracker* storage_tracker = cgrad_storage_start_tracking();
    
    // Create shallow copy of source
    int err = cgrad_storage_shallow_copy(src, dst);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }
    
    // Apply transpose to the layout
    err = cgrad_storage_layout_transpose(dst->backend->storage_get_layout(dst->data), perm, ndim);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
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
    cgrad_storage_registry_tracker* storage_tracker = cgrad_storage_start_tracking();

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
            cgrad_storage_stop_tracking_and_free_all(storage_tracker);
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
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }
    
    // Create ones tensor of shape (summed_size, 1)
    cgrad_storage ones = {0};
    err = cgrad_storage_init(&ones, (const uint32_t[]){summed_size, 1}, 2, a_perm.backend->type);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }
    err = cgrad_storage_fill(&ones, 1.0f);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }
    
    // Compute sum via GEMM
    cgrad_storage r_mat = {0};
    err = cgrad_storage_gemm(&a_reshaped, &ones, &r_mat);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }

    // Reshape r_mat back to target shape
    err = cgrad_storage_reshape(&r_mat, r, target_shape, TENSOR_DIM);
    if (err != CGRAD_SUCCESS) {
        cgrad_storage_stop_tracking_and_free_all(storage_tracker);
        return err;
    }

    // cleanup and do not free output storage
    cgrad_storage_registry_tracker_remove(storage_tracker, r);
    err = cgrad_storage_stop_tracking_and_free_all(storage_tracker);

    return err;
}

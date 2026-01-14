#ifndef CGRAD_STORAGE_H
#define CGRAD_STORAGE_H

#include "cgrad_storage_backend.h"
#include <stdint.h>
#include <stddef.h>
#include <uuid/uuid.h>

/**
 * @brief High-level storage object supporting multiple backends.
 */
typedef struct cgrad_storage {
    uuid_t uuid;                        /**< Unique identifier for this storage */
    cgrad_storage_backend* backend;     /**< Pointer to backend ops */
    void* data;                         /**< Backend-specific storage object (e.g., cgrad_tensor_f32*) */
} cgrad_storage;

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
int cgrad_storage_init(cgrad_storage* t, const uint32_t* shape, int ndim, cgrad_storage_backend_type backend_type);

/**
 * @brief Perform a shallow copy of a tensor (copies data pointer, not underlying data).
 * @param src Source tensor.
 * @param dst Destination tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_shallow_copy(const cgrad_storage* src, cgrad_storage* dst);

/**
 * @brief Make a contiguous copy of a tensor into dst.
 * @param src Source tensor.
 * @param dst Destination tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_contiguous(const cgrad_storage* src, cgrad_storage* dst);

/**
 * @brief Free the memory associated with a high-level tensor.
 *        Returns the error code from the registry deregistration.
 * @param t Pointer to tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_free(cgrad_storage* t);

/**
 * @brief Cleanup the global storage registry.
 * 
 * This should be called at program shutdown to free all registry resources.
 */
void cgrad_storage_cleanup_global_registry(void);

// ============================================================================
// Storage Tracking API (Scoped Resource Management)
// ============================================================================

/**
 * @brief Start tracking storage allocations.
 *        All storages created after this call will be tracked.
 *        Returns a tracker handle that can be used to free all tracked storages.
 * 
 * @return Pointer to tracker, or NULL on allocation failure.
 */
struct cgrad_storage_registry_tracker* cgrad_storage_start_tracking(void);

/**
 * @brief Stop tracking and free all tracked storages.
 *        Frees all storages that were registered since the tracker was started.
 *        If errors occur during freeing, continues to free all storages but returns the first error.
 * 
 * @param tracker Pointer to the tracker from cgrad_storage_start_tracking.
 * @return CGRAD_SUCCESS if all storages freed successfully, otherwise the first error code encountered.
 */
int cgrad_storage_stop_tracking_and_free_all(struct cgrad_storage_registry_tracker* tracker);

/**
 * @brief Fill the tensor with a constant value.
 * @param t Pointer to tensor.
 * @param value The value to fill the tensor with.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_fill(cgrad_storage* t, float value);

/**
 * @brief Fill the tensor with random values.
 * @param t Pointer to tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_fill_rand(cgrad_storage* t);

// --- Math Ops ---

/**
 * @brief Sum a tensor over specified axes using reshape and GEMM with a tensor of all ones.
 * @param a Input tensor.
 * @param mask Right-aligned mask (length ndim) indicating which axes to sum (1=sum, 0=keep).
 * @param ndim Number of dimensions in mask (≤ TENSOR_DIM).
 * @param r Output tensor (initialized inside function).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_sum(const cgrad_storage* a, const uint8_t* mask, int ndim, cgrad_storage* r);

/**
 * @brief Perform batched matrix multiplication (GEMM) on two tensors.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_gemm(const cgrad_storage* a, const cgrad_storage* b, cgrad_storage* r);

/**
 * @brief Add two tensors elementwise and store the result in a third tensor.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */

int cgrad_storage_add(cgrad_storage* a, cgrad_storage* b, cgrad_storage* r);

/**
 * @brief Subtract two tensors elementwise and store the result in a third tensor.
 *        Computes r = a - b.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_sub(cgrad_storage* a, cgrad_storage* b, cgrad_storage* r);

// --- Data Access/Info ---

/**
 * @brief Print the tensor's shape and contents.
 * @param t Pointer to tensor.
 */
void cgrad_storage_print(const cgrad_storage* t);

/**
 * @brief Reshape a tensor, using layout reshape and backend copy ops.
 * @param src Source tensor.
 * @param dst Destination tensor.
 * @param new_shape Array of new dimensions (length ndim, may contain one -1).
 * @param ndim Number of dimensions in new_shape (<= TENSOR_DIM).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_reshape(const cgrad_storage* src, cgrad_storage* dst, const int32_t* new_shape, int ndim);

// --- Transform ---

/**
 * @brief Transpose the tensor according to the given permutation, applied to the last ndim dims.
 * Creates a shallow copy of the source tensor and applies the transpose to the layout.
 * @param src Source tensor.
 * @param dst Destination tensor (will be initialized with shallow copy + transpose).
 * @param perm Permutation array (length ndim).
 * @param ndim Number of trailing dimensions to permute (≤ MAX_TENSOR_DIM).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_transpose(const cgrad_storage* src, cgrad_storage* dst, const uint32_t* perm, int ndim);

#endif // CGRAD_STORAGE_H

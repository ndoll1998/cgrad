#ifndef CGRAD_STORAGE_H
#define CGRAD_STORAGE_H

#include "backends/cgrad_backend.h"
#include <stdint.h>
#include <stddef.h>
#include <uuid/uuid.h>

/**
 * @brief High-level storage object supporting multiple backends.
 */
typedef struct cgrad_storage {
    uuid_t uuid;                        /**< Unique identifier for this storage */
    cgrad_backend* backend;     /**< Pointer to backend ops */
    void* data;                         /**< Backend-specific storage object (e.g., cgrad_tensor_f32*) */
} cgrad_storage;

// --- Initialization/Allocation ---

/**
 * @brief Initialize a high-level tensor with the given shape, ndim, and backend name.
 *        The user-specified shape (length ndim) is placed at the end; leading unspecified dims are set to 1.
 *        For example, shape={3,4}, ndim=2, MAX_TENSOR_DIM=4 => layout.shape={1,1,3,4}
 * @param t Pointer to tensor to initialize.
 * @param shape Array of dimensions (length ndim).
 * @param ndim Number of dimensions in shape (≤ MAX_TENSOR_DIM).
 * @param backend_name Backend name to use (e.g., "cpu_f32").
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_init(cgrad_storage* t, const uint32_t* shape, int ndim, const char* backend_name);

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
 // Storage Recording API (Scoped Resource Management)
 // ============================================================================

/**
 * @brief Start recording storage allocations.
 *        All storages created after this call will be recorded.
 *        Returns a record handle that can be used to free all recorded storages.
 * 
 * @return Pointer to record, or NULL on allocation failure.
 */
struct cgrad_storage_registry_record* cgrad_storage_start_recording(void);

/**
 * @brief Stop recording storage allocations.
 *        The record remains valid and contains all recorded storage UUIDs.
 * 
 * @param record Pointer to the record from cgrad_storage_start_recording.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_stop_recording(struct cgrad_storage_registry_record* record);

/**
 * @brief Free all storages recorded in a record.
 *        Frees all storages that were recorded since the record was started.
 *        If errors occur during freeing, continues to free all storages but returns the first error.
 * 
 * @param record Pointer to the record from cgrad_storage_start_recording.
 * @return CGRAD_SUCCESS if all storages freed successfully, otherwise the first error code encountered.
 */
int cgrad_storage_free_all_from_record(struct cgrad_storage_registry_record* record);

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
 *        Computes r = alpha * a * b + beta * r.
 * @param alpha Scaling factor for the matrix product.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param beta Scaling factor for the output tensor.
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_gemm(float alpha, const cgrad_storage* a, const cgrad_storage* b, float beta, cgrad_storage* r);

/**
 * @brief Compute y = alpha * x + y (AXPY operation).
 *        Computes r = alpha * x + r.
 * @param alpha Scaling factor for x.
 * @param x First input tensor.
 * @param y Second input tensor (used to initialize r if r is uninitialized).
 * @param r Output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_axpy(float alpha, cgrad_storage* x, cgrad_storage* y, cgrad_storage* r);

// --- Data Transform ---

/**
 * @brief Reshape a tensor, using layout reshape and backend copy ops.
 * @param src Source tensor.
 * @param dst Destination tensor.
 * @param new_shape Array of new dimensions (length ndim, may contain one -1).
 * @param ndim Number of dimensions in new_shape (<= TENSOR_DIM).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_reshape(const cgrad_storage* src, cgrad_storage* dst, const int32_t* new_shape, int ndim);


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

// --- Data Access/Info ---

/**
 * @brief Get the value at the given indices.
 * @param t Pointer to storage.
 * @param indices Array of indices.
 * @param ndim Number of dimensions in indices.
 * @param out_value Pointer to float where the value will be written.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_storage_get(const cgrad_storage* t, const uint32_t* indices, int ndim, float* out_value);

/**
 * @brief Print the tensor's shape and contents.
 * @param t Pointer to tensor.
 */
void cgrad_storage_print(const cgrad_storage* t);

#endif // CGRAD_STORAGE_H

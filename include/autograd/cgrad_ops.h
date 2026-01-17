#ifndef CGRAD_OPS_H
#define CGRAD_OPS_H

#include "storage/cgrad_storage.h"
#include "storage/cgrad_storage_layout.h"

/**
 * @file cgrad_ops.h
 * @brief Operation abstraction layer for forward and backward passes.
 * 
 * Each operation implements a forward function and a backward function.
 * The forward function computes the output and optionally caches intermediate
 * results in a context structure. The backward function uses the cached context
 * to compute gradients efficiently.
 */

/**
 * @brief Union containing operation-specific metadata.
 * The relevant field depends on the operation type.
 */
typedef union cgrad_op_metadata {
    struct {
        uint32_t perm[TENSOR_DIM];  /**< Permutation for transpose */
        int ndim;                   /**< Number of dimensions to permute */
    } transpose;
    
    struct {
        int32_t new_shape[TENSOR_DIM];  /**< Target shape for reshape */
        int ndim;                        /**< Number of dimensions */
    } reshape;
    
    struct {
        uint8_t mask[TENSOR_DIM];   /**< Reduction mask (1=reduce, 0=keep) */
        int ndim;                   /**< Number of dimensions */
    } reduce_sum;
    
    struct {
        float alpha;                /**< Scalar multiplier for A*B */
        float beta;                 /**< Scalar multiplier for C */
    } gemm;
    
    struct {
        float alpha;                /**< Scalar multiplier for x in y = alpha*x + y */
    } axpy;
    
    float scalar;                   /**< For scalar operations */
} cgrad_op_metadata;

/**
 * @brief Function pointer type for forward pass operations.
 * 
 * @param inputs Array of input storages.
 * @param num_inputs Number of inputs.
 * @param metadata Operation-specific metadata.
 * @param output Output storage (allocated by caller).
 * @param ctx Pointer to context pointer. Operation can allocate and store intermediate results here.
 * @param requires_grad Whether this operation requires gradient computation (affects context caching).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
typedef int (*cgrad_op_forward_fn)(
    cgrad_storage** inputs,
    int num_inputs,
    const cgrad_op_metadata* metadata,
    cgrad_storage* output,
    void** ctx,
    int requires_grad
);

/**
 * @brief Function pointer type for backward pass operations.
 * 
 * Computes gradients with respect to inputs given the gradient of the output.
 * Gradients are ACCUMULATED into grad_inputs (not overwritten).
 * 
 * @param inputs Array of input storages (from forward pass).
 * @param num_inputs Number of inputs.
 * @param output Output storage (from forward pass).
 * @param grad_output Gradient of the output (incoming gradient).
 * @param metadata Operation-specific metadata.
 * @param ctx Context pointer containing cached intermediate results from forward pass.
 * @param grad_inputs Array of gradient storages for inputs (pre-allocated, accumulate into these).
 * @param input_requires_grad Array indicating which inputs require gradients.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
typedef int (*cgrad_op_backward_fn)(
    cgrad_storage** inputs,
    int num_inputs,
    cgrad_storage* output,
    cgrad_storage* grad_output,
    const cgrad_op_metadata* metadata,
    void* ctx,
    cgrad_storage** grad_inputs,
    const int* input_requires_grad
);

/**
 * @brief Operation descriptor containing forward and backward functions.
 */
typedef struct cgrad_op_descriptor {
    const char* name;                /**< Human-readable name */
    cgrad_op_forward_fn forward;     /**< Forward pass function */
    cgrad_op_backward_fn backward;   /**< Backward pass function */
} cgrad_op_descriptor;

/**
 * @brief Information about an operation and its metadata.
 */
typedef struct cgrad_op_info {
    const cgrad_op_descriptor* descriptor;  /**< Pointer to operation descriptor (NULL for leaf nodes) */
    cgrad_op_metadata metadata;             /**< Operation-specific metadata */
} cgrad_op_info;

// ============================================================================
// Operation Forward/Backward Function Declarations (implemented in src/autograd/ops/*.c)
// ============================================================================

// AXPY operation
int cgrad_op_axpy_forward(
    cgrad_storage** inputs,
    int num_inputs,
    const cgrad_op_metadata* metadata,
    cgrad_storage* output,
    void** ctx,
    int requires_grad
);

int cgrad_op_axpy_backward(
    cgrad_storage** inputs,
    int num_inputs,
    cgrad_storage* output,
    cgrad_storage* grad_output,
    const cgrad_op_metadata* metadata,
    void* ctx,
    cgrad_storage** grad_inputs,
    const int* input_requires_grad
);

// GEMM operation
int cgrad_op_gemm_forward(
    cgrad_storage** inputs,
    int num_inputs,
    const cgrad_op_metadata* metadata,
    cgrad_storage* output,
    void** ctx,
    int requires_grad
);

int cgrad_op_gemm_backward(
    cgrad_storage** inputs,
    int num_inputs,
    cgrad_storage* output,
    cgrad_storage* grad_output,
    const cgrad_op_metadata* metadata,
    void* ctx,
    cgrad_storage** grad_inputs,
    const int* input_requires_grad
);

// Transpose operation
int cgrad_op_transpose_forward(
    cgrad_storage** inputs,
    int num_inputs,
    const cgrad_op_metadata* metadata,
    cgrad_storage* output,
    void** ctx,
    int requires_grad
);

int cgrad_op_transpose_backward(
    cgrad_storage** inputs,
    int num_inputs,
    cgrad_storage* output,
    cgrad_storage* grad_output,
    const cgrad_op_metadata* metadata,
    void* ctx,
    cgrad_storage** grad_inputs,
    const int* input_requires_grad
);

// Reshape operation
int cgrad_op_reshape_forward(
    cgrad_storage** inputs,
    int num_inputs,
    const cgrad_op_metadata* metadata,
    cgrad_storage* output,
    void** ctx,
    int requires_grad
);

int cgrad_op_reshape_backward(
    cgrad_storage** inputs,
    int num_inputs,
    cgrad_storage* output,
    cgrad_storage* grad_output,
    const cgrad_op_metadata* metadata,
    void* ctx,
    cgrad_storage** grad_inputs,
    const int* input_requires_grad
);

// Reduce sum operation
int cgrad_op_reduce_sum_forward(
    cgrad_storage** inputs,
    int num_inputs,
    const cgrad_op_metadata* metadata,
    cgrad_storage* output,
    void** ctx,
    int requires_grad
);

int cgrad_op_reduce_sum_backward(
    cgrad_storage** inputs,
    int num_inputs,
    cgrad_storage* output,
    cgrad_storage* grad_output,
    const cgrad_op_metadata* metadata,
    void* ctx,
    cgrad_storage** grad_inputs,
    const int* input_requires_grad
);

// ============================================================================
// Operation Descriptors (defined inline)
// ============================================================================

static const cgrad_op_descriptor cgrad_op_axpy = {
    .name = "AXPY",
    .forward = cgrad_op_axpy_forward,
    .backward = cgrad_op_axpy_backward
};

static const cgrad_op_descriptor cgrad_op_gemm = {
    .name = "GEMM",
    .forward = cgrad_op_gemm_forward,
    .backward = cgrad_op_gemm_backward
};

static const cgrad_op_descriptor cgrad_op_transpose = {
    .name = "TRANSPOSE",
    .forward = cgrad_op_transpose_forward,
    .backward = cgrad_op_transpose_backward
};

static const cgrad_op_descriptor cgrad_op_reshape = {
    .name = "RESHAPE",
    .forward = cgrad_op_reshape_forward,
    .backward = cgrad_op_reshape_backward
};

static const cgrad_op_descriptor cgrad_op_reduce_sum = {
    .name = "REDUCE_SUM",
    .forward = cgrad_op_reduce_sum_forward,
    .backward = cgrad_op_reduce_sum_backward
};

#endif // CGRAD_OPS_H

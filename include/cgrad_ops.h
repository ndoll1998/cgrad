#ifndef CGRAD_OPS_H
#define CGRAD_OPS_H

#include "cgrad_storage.h"
#include "cgrad_compute_graph.h"

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
 * @brief Function pointer type for forward pass operations.
 * 
 * @param inputs Array of input storages.
 * @param num_inputs Number of inputs.
 * @param metadata Operation-specific metadata.
 * @param output Output storage (allocated by caller).
 * @param ctx Pointer to context pointer. Operation can allocate and store intermediate results here.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
typedef int (*cgrad_op_forward_fn)(
    cgrad_storage** inputs,
    int num_inputs,
    const cgrad_op_metadata* metadata,
    cgrad_storage* output,
    void** ctx
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
 * @brief Function pointer type for freeing operation context.
 * 
 * @param ctx Context pointer to free.
 */
typedef void (*cgrad_op_free_ctx_fn)(void* ctx);

/**
 * @brief Operation descriptor containing forward and backward functions.
 */
typedef struct {
    cgrad_op_type type;              /**< Operation type */
    const char* name;                /**< Human-readable name */
    cgrad_op_forward_fn forward;     /**< Forward pass function */
    cgrad_op_backward_fn backward;   /**< Backward pass function */
    cgrad_op_free_ctx_fn free_ctx;   /**< Context cleanup function (NULL if no context) */
} cgrad_op_descriptor;

/**
 * @brief Get the operation descriptor for a given operation type.
 * 
 * @param op_type Operation type.
 * @return Pointer to operation descriptor, or NULL if not found.
 */
const cgrad_op_descriptor* cgrad_get_op_descriptor(cgrad_op_type op_type);

// ============================================================================
// Operation Descriptors (defined in src/ops/*.c)
// ============================================================================

extern const cgrad_op_descriptor cgrad_op_add_descriptor;
extern const cgrad_op_descriptor cgrad_op_sub_descriptor;
extern const cgrad_op_descriptor cgrad_op_gemm_descriptor;
extern const cgrad_op_descriptor cgrad_op_transpose_descriptor;
extern const cgrad_op_descriptor cgrad_op_reshape_descriptor;
extern const cgrad_op_descriptor cgrad_op_reduce_sum_descriptor;

#endif // CGRAD_OPS_H

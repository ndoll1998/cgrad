#include "autograd/cgrad_ops.h"
#include "cgrad_status.h"
#include "storage/cgrad_storage.h"
#include "storage/cgrad_storage_layout.h"

/**
 * @brief Forward pass for reduce sum.
 * 
 * Computes: output = sum(input, mask)
 * No context is needed for backward pass.
 */
int cgrad_op_reduce_sum_forward(
    cgrad_storage** inputs,
    int num_inputs,
    const cgrad_op_metadata* metadata,
    cgrad_storage* output,
    void** ctx,
    int requires_grad
) {
    (void)requires_grad;  // Unused for now
    if (num_inputs != 1) {
        return CGRAD_ERR_COMPUTE_GRAPH_INVALID_OPERATION;
    }
    
    *ctx = NULL;
    
    return cgrad_storage_reduce(
        1.0f,
        inputs[0],
        metadata->reduce_sum.mask,
        metadata->reduce_sum.ndim,
        0.0f,
        output
    );
}

/**
 * @brief Backward pass for reduce sum.
 * 
 * For B = sum(A, mask):
 *   grad_A += broadcast(grad_B, original_shape)
 * 
 * We need to broadcast the gradient back to the original shape.
 * Since grad_output has reduced shape and grad_inputs[0] has original shape,
 * we need to add grad_output (broadcast) to grad_inputs[0].
 */
int cgrad_op_reduce_sum_backward(
    cgrad_storage** inputs,
    int num_inputs,
    cgrad_storage* output,
    cgrad_storage* grad_output,
    const cgrad_op_metadata* metadata,
    void* ctx,
    cgrad_storage** grad_inputs,
    const int* input_requires_grad
) {
    if (num_inputs != 1) {
        return CGRAD_ERR_COMPUTE_GRAPH_INVALID_OPERATION;
    }
    
    if (!input_requires_grad[0]) {
        return CGRAD_SUCCESS;
    }
    
    return cgrad_storage_axpy(1.0f, grad_output, grad_inputs[0], grad_inputs[0]);
}

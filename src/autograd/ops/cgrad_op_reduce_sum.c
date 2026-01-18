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
    
    if (!input_requires_grad[0] || grad_inputs[0] == NULL) {
        return CGRAD_SUCCESS;
    }
    
    // For reduce_sum backward, we need to broadcast grad_output to the original shape
    // and add it to grad_inputs[0].
    // 
    // The grad_output has shape with 1s where we summed, and grad_inputs[0] has the
    // original shape. We use storage_add which broadcasts automatically.
    //
    // However, storage_add doesn't support writing to existing tensor yet.
    // So we manually broadcast by using the backend's storage_add with broadcasting.
    
    // Create a shallow copy of grad_output for broadcasting
    cgrad_storage grad_out_bcast;
    int err = cgrad_storage_shallow_copy(grad_output, &grad_out_bcast);
    if (err != CGRAD_SUCCESS) return err;
    
    // Broadcast the layout to match grad_inputs[0]
    cgrad_storage_layout* grad_out_layout = grad_out_bcast.backend->storage_get_layout(grad_out_bcast.data);
    cgrad_storage_layout* grad_in_layout = grad_inputs[0]->backend->storage_get_layout(grad_inputs[0]->data);
    
    err = cgrad_storage_layout_broadcast(grad_out_layout, grad_in_layout, 0, TENSOR_DIM);
    if (err != CGRAD_SUCCESS) return err;
    
    // Now add: grad_inputs[0] += grad_out_bcast
    err = grad_inputs[0]->backend->storage_axpy(
        1.0f,
        grad_out_bcast.data,
        grad_inputs[0]->data
    );
    
    return err;
}

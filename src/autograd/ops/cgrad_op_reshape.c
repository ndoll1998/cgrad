#include "autograd/cgrad_ops.h"
#include "cgrad_status.h"
#include "storage/cgrad_storage.h"
#include "storage/cgrad_storage_layout.h"

/**
 * @brief Forward pass for reshape.
 * 
 * Computes: output = reshape(input, new_shape)
 * No context is needed for backward pass.
 */
int cgrad_op_reshape_forward(
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
    
    return cgrad_storage_reshape(
        inputs[0],
        output,
        metadata->reshape.new_shape,
        metadata->reshape.ndim
    );
}

/**
 * @brief Backward pass for reshape.
 * 
 * For B = reshape(A, new_shape):
 *   grad_A += reshape(grad_B, original_shape)
 */
int cgrad_op_reshape_backward(
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
    
    int ret;
    
    // Get original shape from input storage
    cgrad_storage_layout* input_layout = inputs[0]->backend->storage_get_layout(inputs[0]->data);
    
    int32_t orig_shape[TENSOR_DIM];
    for (int k = 0; k < TENSOR_DIM; k++) {
        orig_shape[k] = (int32_t)input_layout->shape[k];
    }
    
    // Reshape gradient back to original shape
    cgrad_storage grad_input = {0};
    ret = cgrad_storage_reshape(grad_output, &grad_input, orig_shape, TENSOR_DIM);
    if (ret != CGRAD_SUCCESS) return ret;
    
    // Accumulate: grad_A = grad_A + grad_input
    ret = grad_inputs[0]->backend->storage_axpy(
        1.0f,
        grad_input.data,
        grad_inputs[0]->data
    );
    cgrad_storage_free(&grad_input);
    
    return ret;
}


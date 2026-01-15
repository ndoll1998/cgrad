#include "autograd/cgrad_ops.h"
#include "cgrad_errors.h"
#include "storage/cgrad_storage.h"

/**
 * @brief Forward pass for element-wise addition.
 * 
 * Computes: output = a + b
 * No context is needed for backward pass.
 */
static int add_forward(
    cgrad_storage** inputs,
    int num_inputs,
    const cgrad_op_metadata* metadata,
    cgrad_storage* output,
    void** ctx
) {
    if (num_inputs != 2) {
        return CGRAD_GRAPH_ERR_INVALID_OPERATION;
    }
    
    // Addition doesn't need to cache anything for backward pass
    *ctx = NULL;
    
    return cgrad_storage_add(1.0f, inputs[0], inputs[1], output);
}

/**
 * @brief Backward pass for element-wise addition.
 * 
 * For c = a + b:
 *   grad_a += grad_output
 *   grad_b += grad_output
 */
static int add_backward(
    cgrad_storage** inputs,
    int num_inputs,
    cgrad_storage* output,
    cgrad_storage* grad_output,
    const cgrad_op_metadata* metadata,
    void* ctx,
    cgrad_storage** grad_inputs,
    const int* input_requires_grad
) {
    if (num_inputs != 2) {
        return CGRAD_GRAPH_ERR_INVALID_OPERATION;
    }
    
    int ret;
    
    // Gradient for input 0: grad_a += grad_output
    if (input_requires_grad[0] && grad_inputs[0] != NULL) {
        ret = grad_inputs[0]->backend->storage_add(
            1.0f,
            grad_output->data,
            grad_inputs[0]->data,
            grad_inputs[0]->data
        );
        if (ret != CGRAD_SUCCESS) return ret;
    }
    
    // Gradient for input 1: grad_b += grad_output
    if (input_requires_grad[1] && grad_inputs[1] != NULL) {
        ret = grad_inputs[1]->backend->storage_add(
            1.0f,
            grad_output->data,
            grad_inputs[1]->data,
            grad_inputs[1]->data
        );
        if (ret != CGRAD_SUCCESS) return ret;
    }
    
    return CGRAD_SUCCESS;
}

const cgrad_op_descriptor cgrad_op_add_descriptor = {
    .type = CGRAD_OP_ADD,
    .name = "ADD",
    .forward = add_forward,
    .backward = add_backward,
    .free_ctx = NULL
};

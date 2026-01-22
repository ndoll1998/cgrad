#include "autograd/cgrad_ops.h"
#include "cgrad_status.h"
#include "storage/cgrad_storage.h"
#include "storage/cgrad_storage_registry.h"

/**
 * @brief Helper function to update gradient with optional broadcasting reduction.
 * 
 * This function handles gradient accumulation for a single input, automatically
 * detecting if broadcasting occurred and reducing the gradient accordingly.
 * 
 * @param alpha Scaling factor for grad_output
 * @param grad_output Gradient from the output
 * @param grad_input Gradient accumulator for the input (modified in-place)
 * @return CGRAD_SUCCESS on success, error code otherwise
 */
static cgrad_status cgrad_op_axpy_update_gradient(
    float alpha,
    cgrad_storage* grad_output,
    cgrad_storage* grad_input
) {
    const cgrad_storage_layout* grad_out_layout = grad_output->backend->storage_get_layout(grad_output->data);
    const cgrad_storage_layout* grad_in_layout = grad_input->backend->storage_get_layout(grad_input->data);
    
    // Detect which dimensions were broadcasted
    uint8_t reduction_mask[TENSOR_DIM] = {0};
    int needs_reduction = 0;
    for (int d = 0; d < TENSOR_DIM; d++) {
        if (grad_out_layout->shape[d] != grad_in_layout->shape[d]) {
            reduction_mask[d] = 1;
            needs_reduction = 1;
        }
    }
    
    cgrad_storage tmp = {0};
    cgrad_storage_registry_record* storage_record = cgrad_storage_start_recording();
    
    if (needs_reduction) {
        // Broadcasting occurred: need to reduce grad_output
        cgrad_status err = cgrad_storage_reduce(1.0, grad_output, reduction_mask, TENSOR_DIM, 0.0f, &tmp);
        if (err != CGRAD_SUCCESS) {
            cgrad_storage_stop_recording(storage_record);
            cgrad_storage_free_record(storage_record);
            return err;
        }
        // use reduced gradient output
        grad_output = &tmp;
    }
    
    // No broadcasting: accumulate directly (most efficient)
    cgrad_status err = cgrad_storage_axpy(alpha, grad_output, grad_input, grad_input);
    
    // cleanup
    cgrad_storage_stop_recording(storage_record);
    cgrad_storage_free_record(storage_record);

    return err;
}

/**
 * @brief Forward pass for AXPY operation (y = alpha * x + y).
 * 
 * Computes: output = alpha * x + y
 * No context is needed for backward pass.
 */
cgrad_status cgrad_op_axpy_forward(
    cgrad_storage** inputs,
    int num_inputs,
    const cgrad_op_metadata* metadata,
    cgrad_storage* output,
    void** ctx,
    int requires_grad
) {
    (void)requires_grad;  // Unused for now
    if (num_inputs != 2) {
        return CGRAD_ERR_COMPUTE_GRAPH_INVALID_OPERATION;
    }
    
    // AXPY doesn't need to cache anything for backward pass
    *ctx = NULL;
    
    // Use alpha from metadata
    float alpha = metadata->axpy.alpha;
    
    return cgrad_storage_axpy(alpha, inputs[0], inputs[1], output);
}

/**
 * @brief Backward pass for AXPY operation with broadcasting support.
 * 
 * For c = alpha * a + b:
 *   grad_a += alpha * grad_output (reduced if a was broadcasted)
 *   grad_b += grad_output (reduced if b was broadcasted)
 * 
 * When broadcasting occurred in the forward pass, we must sum the gradient
 * across the broadcasted dimensions to match the original input shape.
 */
cgrad_status cgrad_op_axpy_backward(
    cgrad_storage** inputs,
    int num_inputs,
    cgrad_storage* output,
    cgrad_storage* grad_output,
    const cgrad_op_metadata* metadata,
    void* ctx,
    cgrad_storage** grad_inputs,
    const int* input_requires_grad
) {
    (void)ctx;
    (void)output;
    
    if (num_inputs != 2) {
        return CGRAD_ERR_COMPUTE_GRAPH_INVALID_OPERATION;
    }
    
    int ret;
    float alpha = metadata->axpy.alpha;
    
    // Process gradient for input 0: grad_x += alpha * grad_output
    if (input_requires_grad[0] && grad_inputs[0] != NULL) {
        ret = cgrad_op_axpy_update_gradient(alpha, grad_output, grad_inputs[0]);
        if (ret != CGRAD_SUCCESS) return ret;
    }
    
    // Process gradient for input 1: grad_y += grad_output
    if (input_requires_grad[1] && grad_inputs[1] != NULL) {
        ret = cgrad_op_axpy_update_gradient(1.0f, grad_output, grad_inputs[1]);
        if (ret != CGRAD_SUCCESS) return ret;
    }
    
    return CGRAD_SUCCESS;
}

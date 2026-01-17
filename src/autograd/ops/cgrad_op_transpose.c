#include "autograd/cgrad_ops.h"
#include "cgrad_status.h"
#include "storage/cgrad_storage.h"
#include "storage/cgrad_storage_layout.h"

/**
 * @brief Forward pass for transpose.
 * 
 * Computes: output = transpose(input, perm)
 * No context is needed for backward pass.
 */
static int transpose_forward(
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
    
    return cgrad_storage_transpose(
        inputs[0],
        output,
        metadata->transpose.perm,
        metadata->transpose.ndim
    );
}

/**
 * @brief Backward pass for transpose.
 * 
 * For B = transpose(A, perm):
 *   grad_A += transpose(grad_B, inverse_perm)
 */
static int transpose_backward(
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
    int ndim = metadata->transpose.ndim;
    
    // Compute inverse permutation
    uint32_t inv_perm[TENSOR_DIM];
    for (int k = 0; k < ndim; k++) {
        inv_perm[metadata->transpose.perm[k]] = k;
    }
    
    // Transpose gradient back
    cgrad_storage grad_input = {0};
    ret = cgrad_storage_transpose(grad_output, &grad_input, inv_perm, ndim);
    if (ret != CGRAD_SUCCESS) return ret;
    
    // Accumulate: grad_A = grad_A + grad_input
    ret = grad_inputs[0]->backend->storage_axpy(
        1.0f,
        grad_input.data,
        grad_inputs[0]->data,
        grad_inputs[0]->data
    );
    cgrad_storage_free(&grad_input);
    
    return ret;
}

const cgrad_op_descriptor cgrad_op_transpose_descriptor = {
    .type = CGRAD_OP_TRANSPOSE,
    .name = "TRANSPOSE",
    .forward = transpose_forward,
    .backward = transpose_backward,
    .free_ctx = NULL
};

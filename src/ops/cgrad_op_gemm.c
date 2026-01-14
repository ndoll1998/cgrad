#include "cgrad_ops.h"
#include "cgrad_errors.h"
#include "cgrad_storage.h"
#include <stdlib.h>

/**
 * @brief Forward pass for matrix multiplication (GEMM).
 * 
 * Computes: output = A @ B
 * No context is needed - we use input storages directly in backward.
 */
static int gemm_forward(
    cgrad_storage** inputs,
    int num_inputs,
    const cgrad_op_metadata* metadata,
    cgrad_storage* output,
    void** ctx
) {
    if (num_inputs != 2) {
        return CGRAD_GRAPH_ERR_INVALID_OPERATION;
    }
    
    *ctx = NULL;
    
    return cgrad_storage_gemm(inputs[0], inputs[1], output);
}

/**
 * @brief Backward pass for matrix multiplication (GEMM).
 * 
 * For C = A @ B:
 *   grad_A += grad_C @ B^T
 *   grad_B += A^T @ grad_C
 */
static int gemm_backward(
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
    uint32_t perm[] = {1, 0};
    
    // Gradient for input 0 (A): grad_A += grad_C @ B^T
    if (input_requires_grad[0] && grad_inputs[0] != NULL) {
        cgrad_storage b_transposed = {0};
        ret = cgrad_storage_transpose(inputs[1], &b_transposed, perm, 2);
        if (ret != CGRAD_SUCCESS) return ret;
        
        cgrad_storage grad_a_contrib = {0};
        ret = cgrad_storage_gemm(grad_output, &b_transposed, &grad_a_contrib);
        cgrad_storage_free(&b_transposed);
        if (ret != CGRAD_SUCCESS) return ret;
        
        // Accumulate: grad_A = grad_A + grad_a_contrib
        ret = grad_inputs[0]->backend->storage_add(
            1.0f,
            grad_a_contrib.data,
            grad_inputs[0]->data,
            grad_inputs[0]->data
        );
        cgrad_storage_free(&grad_a_contrib);
        if (ret != CGRAD_SUCCESS) return ret;
    }
    
    // Gradient for input 1 (B): grad_B += A^T @ grad_C
    if (input_requires_grad[1] && grad_inputs[1] != NULL) {
        cgrad_storage a_transposed = {0};
        ret = cgrad_storage_transpose(inputs[0], &a_transposed, perm, 2);
        if (ret != CGRAD_SUCCESS) return ret;
        
        cgrad_storage grad_b_contrib = {0};
        ret = cgrad_storage_gemm(&a_transposed, grad_output, &grad_b_contrib);
        cgrad_storage_free(&a_transposed);
        if (ret != CGRAD_SUCCESS) return ret;
        
        // Accumulate: grad_B = grad_B + grad_b_contrib
        ret = grad_inputs[1]->backend->storage_add(
            1.0f,
            grad_b_contrib.data,
            grad_inputs[1]->data,
            grad_inputs[1]->data
        );
        cgrad_storage_free(&grad_b_contrib);
        if (ret != CGRAD_SUCCESS) return ret;
    }
    
    return CGRAD_SUCCESS;
}

const cgrad_op_descriptor cgrad_op_gemm_descriptor = {
    .type = CGRAD_OP_GEMM,
    .name = "GEMM",
    .forward = gemm_forward,
    .backward = gemm_backward,
    .free_ctx = NULL
};

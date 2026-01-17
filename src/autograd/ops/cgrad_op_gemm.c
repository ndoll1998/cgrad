#include "autograd/cgrad_ops.h"
#include "cgrad_status.h"
#include "storage/cgrad_storage.h"
#include <stdlib.h>

/**
 * @brief Forward pass for matrix multiplication (GEMM).
 * 
 * Computes: output = A @ B
 * No context is needed - we use input storages directly in backward.
 */
int cgrad_op_gemm_forward(
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
    
    *ctx = NULL;
    
    // Use alpha and beta from metadata
    float alpha = metadata->gemm.alpha;
    float beta = metadata->gemm.beta;
    
    return cgrad_storage_gemm(alpha, inputs[0], inputs[1], beta, output);
}

/**
 * @brief Backward pass for matrix multiplication (GEMM).
 * 
 * For C = A @ B:
 *   grad_A += grad_C @ B^T
 *   grad_B += A^T @ grad_C
 */
int cgrad_op_gemm_backward(
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
        return CGRAD_ERR_COMPUTE_GRAPH_INVALID_OPERATION;
    }
    
    int ret;
    uint32_t perm[] = {1, 0};
    
    // Gradient for input 0 (A): grad_A += grad_C @ B^T
    if (input_requires_grad[0] && grad_inputs[0] != NULL) {
        cgrad_storage b_transposed = {0};
        ret = cgrad_storage_transpose(inputs[1], &b_transposed, perm, 2);
        if (ret != CGRAD_SUCCESS) return ret;
        
        // Accumulate directly: grad_A = grad_A + 1.0 * (grad_C @ B^T)
        ret = cgrad_storage_gemm(1.0f, grad_output, &b_transposed, 1.0f, grad_inputs[0]);
        cgrad_storage_free(&b_transposed);
        if (ret != CGRAD_SUCCESS) return ret;
    }
    
    // Gradient for input 1 (B): grad_B += A^T @ grad_C
    if (input_requires_grad[1] && grad_inputs[1] != NULL) {
        cgrad_storage a_transposed = {0};
        ret = cgrad_storage_transpose(inputs[0], &a_transposed, perm, 2);
        if (ret != CGRAD_SUCCESS) return ret;
        
        // Accumulate directly: grad_B = grad_B + 1.0 * (A^T @ grad_C)
        ret = cgrad_storage_gemm(1.0f, &a_transposed, grad_output, 1.0f, grad_inputs[1]);
        cgrad_storage_free(&a_transposed);
        if (ret != CGRAD_SUCCESS) return ret;
    }
    
    return CGRAD_SUCCESS;
}

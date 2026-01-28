#include "autograd/cgrad_ops.h"
#include "cgrad_status.h"
#include "storage/cgrad_storage.h"
#include "storage/cgrad_storage_registry.h"
#include <stdlib.h>
#include <stdio.h>

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
 * @brief Helper function to compute gradient contribution and accumulate with optional broadcasting reduction.
 * 
 * This function computes the gradient contribution via GEMM and handles gradient accumulation,
 * automatically detecting if broadcasting occurred and reducing the gradient accordingly.
 * When no broadcasting is used, the GEMM directly accumulates into the gradient (beta=1.0).
 * 
 * @param alpha Scalar multiplier for the GEMM operation
 * @param lhs Left-hand side storage for GEMM
 * @param rhs Right-hand side storage for GEMM
 * @param grad_input Gradient accumulator for the input (modified in-place)
 * @return CGRAD_SUCCESS on success, error code otherwise
 */
static cgrad_status cgrad_op_gemm_compute_and_accumulate_gradient(
    float alpha,
    cgrad_storage* lhs,
    cgrad_storage* rhs,
    cgrad_storage* grad_input
) {
    const cgrad_storage_layout* lhs_layout = lhs->backend->storage_get_layout(lhs->data);
    const cgrad_storage_layout* rhs_layout = rhs->backend->storage_get_layout(rhs->data);
    const cgrad_storage_layout* grad_in_layout = grad_input->backend->storage_get_layout(grad_input->data);
    
    // Detect which dimensions would be broadcasted in the GEMM result
    uint8_t reduction_mask[TENSOR_DIM] = {0};
    int needs_reduction = 0;
    
    // Check batch dimensions for broadcasting
    for (int d = 0; d < TENSOR_DIM - 2; d++) {
        uint32_t result_dim = (lhs_layout->shape[d] > rhs_layout->shape[d]) ? lhs_layout->shape[d] : rhs_layout->shape[d];
        if (result_dim != grad_in_layout->shape[d]) {
            reduction_mask[d] = 1;
            needs_reduction = 1;
        }
    }
    
    cgrad_storage_registry_record* storage_record = cgrad_storage_start_recording();
    cgrad_status err;
    
    if (needs_reduction) {
        // Broadcasting occurred: compute into temporary, reduce, then accumulate
        cgrad_storage grad_contrib = {0};
        err = cgrad_storage_gemm(alpha, lhs, rhs, 0.0f, &grad_contrib);
        if (err != CGRAD_SUCCESS) {
            cgrad_storage_stop_recording(storage_record);
            cgrad_storage_free_record(storage_record);
            return err;
        }
        
        cgrad_storage grad_reduced = {0};
        err = cgrad_storage_reduce(1.0, &grad_contrib, reduction_mask, TENSOR_DIM, 0.0f, &grad_reduced);
        if (err != CGRAD_SUCCESS) {
            cgrad_storage_stop_recording(storage_record);
            cgrad_storage_free_record(storage_record);
            return err;
        }
        
        // Accumulate reduced gradient
        err = cgrad_storage_axpy(1.0f, &grad_reduced, grad_input, grad_input);
    } else {
        // No broadcasting: directly accumulate into gradient (beta=1.0)
        err = cgrad_storage_gemm(alpha, lhs, rhs, 1.0f, grad_input);
    }
    
    // cleanup
    cgrad_storage_stop_recording(storage_record);
    cgrad_storage_free_record(storage_record);
    
    return err;
}

/**
 * @brief Backward pass for matrix multiplication (GEMM) with broadcasting support.
 * 
 * For C = A @ B:
 *   grad_A += grad_C @ B^T (reduced if A was broadcasted)
 *   grad_B += A^T @ grad_C (reduced if B was broadcasted)
 * 
 * When broadcasting occurred in the forward pass, we must sum the gradient
 * across the broadcasted dimensions to match the original input shape.
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
    (void)output;
    (void)metadata;
    (void)ctx;
    
    if (num_inputs != 2) {
        return CGRAD_ERR_COMPUTE_GRAPH_INVALID_OPERATION;
    }
    
    int ret;

    // Gradient for input 0 (A): grad_A += grad_C @ B^T
    if (input_requires_grad[0] && grad_inputs[0] != NULL) {
        cgrad_storage_registry_record* storage_record = cgrad_storage_start_recording();
        
        // Simply transpose the last 2 dims of B, then do batched GEMM
        uint32_t perm[] = {1, 0};
        cgrad_storage b_transposed = {0};
        ret = cgrad_storage_transpose(inputs[1], &b_transposed, perm, 2);
        if (ret != CGRAD_SUCCESS) {
            cgrad_storage_stop_recording(storage_record);
            cgrad_storage_free_record(storage_record);
            return ret;
        }
        
        // Compute and accumulate gradient: grad_A += grad_C @ B^T
        ret = cgrad_op_gemm_compute_and_accumulate_gradient(1.0f, grad_output, &b_transposed, grad_inputs[0]);
        
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_record(storage_record);
        
        if (ret != CGRAD_SUCCESS) return ret;
    }
    
    // Gradient for input 1 (B): grad_B += A^T @ grad_C
    if (input_requires_grad[1] && grad_inputs[1] != NULL) {
        cgrad_storage_registry_record* storage_record = cgrad_storage_start_recording();
        
        // Simply transpose the last 2 dims of A, then do batched GEMM
        uint32_t perm[] = {1, 0};
        cgrad_storage a_transposed = {0};
        ret = cgrad_storage_transpose(inputs[0], &a_transposed, perm, 2);
        if (ret != CGRAD_SUCCESS) {
            cgrad_storage_stop_recording(storage_record);
            cgrad_storage_free_record(storage_record);
            return ret;
        }
        
        // Compute and accumulate gradient: grad_B += A^T @ grad_C
        ret = cgrad_op_gemm_compute_and_accumulate_gradient(1.0f, &a_transposed, grad_output, grad_inputs[1]);
        
        cgrad_storage_stop_recording(storage_record);
        cgrad_storage_free_record(storage_record);
        
        if (ret != CGRAD_SUCCESS) return ret;
    }
    
    return CGRAD_SUCCESS;
}

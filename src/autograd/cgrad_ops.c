#include "autograd/cgrad_ops.h"
#include <stddef.h>

/**
 * @file cgrad_ops.c
 * @brief Operation registry - maps operation types to their descriptors.
 */

// Array of all operation descriptors
static const cgrad_op_descriptor* op_descriptors[] = {
    &cgrad_op_axpy_descriptor,
    &cgrad_op_gemm_descriptor,
    &cgrad_op_transpose_descriptor,
    &cgrad_op_reshape_descriptor,
    &cgrad_op_reduce_sum_descriptor,
    NULL  // Sentinel
};

/**
 * @brief Get the operation descriptor for a given operation type.
 * 
 * @param op_type Operation type.
 * @return Pointer to operation descriptor, or NULL if not found.
 */
const cgrad_op_descriptor* cgrad_get_op_descriptor(cgrad_op_type op_type) {
    // CGRAD_OP_NONE has no descriptor
    if (op_type == CGRAD_OP_NONE) {
        return NULL;
    }
    
    // Search through registered descriptors
    for (int i = 0; op_descriptors[i] != NULL; i++) {
        if (op_descriptors[i]->type == op_type) {
            return op_descriptors[i];
        }
    }
    
    return NULL;
}

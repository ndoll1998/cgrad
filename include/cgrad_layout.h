#ifndef CGRAD_LAYOUT_H
#define CGRAD_LAYOUT_H

#include <stdint.h>
#include <stddef.h>

#define MAX_TENSOR_DIM 4

typedef struct cgrad_tensor_layout {
  uint32_t size;
  uint32_t shape[MAX_TENSOR_DIM];
  uint32_t strides[MAX_TENSOR_DIM];
} cgrad_tensor_layout;

/**
 * Deep copy a tensor layout from src to dst.
 * Copies size, shape, and strides.
 */
void cgrad_tensor_layout_copy(cgrad_tensor_layout* dst, const cgrad_tensor_layout* src);

// Layout initialization
int cgrad_tensor_layout_init(cgrad_tensor_layout* l, const uint32_t* shape);

// Indexing
size_t cgrad_tensor_flat_index(const uint32_t* indices, const uint32_t* strides);

/**
 * Broadcast two layouts between start_dim (inclusive) and end_dim (exclusive).
 * Modifies the layouts in-place to broadcast their shapes and strides.
 * 
 * Broadcasting rules:
 * - If the dimensions at a specific index are the same, do nothing.
 * - If they differ and one is 1, set its stride to 0 and set its shape to the other's shape.
 * - If they differ and neither is 1, return err (cannot broadcast).
 * 
 * After broadcasting, the shapes between the specified dimensions will be the same.
 * Returns 0 on success, -1 on failure.
 */
int cgrad_tensor_layout_broadcast(
    cgrad_tensor_layout* l1,
    cgrad_tensor_layout* l2,
    int start_dim,
    int end_dim
);

// Transpose: perm is an array of length MAX_TENSOR_DIM, giving the new order of axes
void cgrad_tensor_layout_transpose(cgrad_tensor_layout* layout, const uint32_t* perm);

// Returns 1 if the layout is contiguous, 0 otherwise
int cgrad_tensor_layout_is_contiguous(const cgrad_tensor_layout* l);

#endif // CGRAD_LAYOUT_H

#include "cgrad_layout.h"

// Layout/tensor initialization
int cgrad_tensor_layout_init(cgrad_tensor_layout* l, const uint32_t* shape) {
  uint32_t cur_stride = 1;
  # pragma unroll
  for (int i = MAX_TENSOR_DIM - 1; i > -1; i--) {
      l->strides[i] = cur_stride;
      l->shape[i] = shape[i];
      cur_stride *= shape[i];
  }
  l->size = cur_stride;
  return 0;
}

/**
 * Deep copy a tensor layout from src to dst.
 * Copies size, shape, and strides.
 */
void cgrad_tensor_layout_copy(cgrad_tensor_layout* dst, const cgrad_tensor_layout* src) {
    if (!dst || !src) return;
    dst->size = src->size;
    for (int i = 0; i < MAX_TENSOR_DIM; ++i) {
        dst->shape[i] = src->shape[i];
        dst->strides[i] = src->strides[i];
    }
}

/**
 * Broadcast two layouts between start_dim (inclusive) and end_dim (exclusive).
 * Modifies the layouts in-place to broadcast their shapes and strides.
 * 
 * Broadcasting rules:
 * - If the dimensions at a specific index are the same, do nothing.
 * - If they differ and one is 1, set its stride to 0 and set its shape to the other's shape.
 * - If they differ and neither is 1, return -1 (cannot broadcast).
 * 
 * After broadcasting, the shapes between the specified dimensions will be the same.
 * Returns 0 on success, -1 on failure.
 */
int cgrad_tensor_layout_broadcast(
    cgrad_tensor_layout* l1,
    cgrad_tensor_layout* l2,
    int start_dim,
    int end_dim
) {
    if (!l1 || !l2) return -1;
    if (start_dim < 0 || end_dim > MAX_TENSOR_DIM || start_dim >= end_dim) return -1;

    for (int i = start_dim; i < end_dim; ++i) {
        uint32_t s1 = l1->shape[i];
        uint32_t s2 = l2->shape[i];
        if (s1 == s2) {
            continue;
        } else if (s1 == 1) {
            l1->shape[i] = s2;
            l1->strides[i] = 0;
        } else if (s2 == 1) {
            l2->shape[i] = s1;
            l2->strides[i] = 0;
        } else {
            return -1;
        }
    }
    return 0;
}

// Indexing
size_t cgrad_tensor_flat_index(const uint32_t* indices, const uint32_t* strides) {
  size_t idx = 0;
  for (int i = 0; i < MAX_TENSOR_DIM; i++) {
    idx += indices[i] * strides[i];
  }
  return idx;
}

void cgrad_tensor_layout_transpose(cgrad_tensor_layout* layout, const uint32_t* perm) {
  uint32_t new_shape[MAX_TENSOR_DIM];
  uint32_t new_strides[MAX_TENSOR_DIM];
  for (int i = 0; i < MAX_TENSOR_DIM; i++) {
    new_shape[i] = layout->shape[perm[i]];
    new_strides[i] = layout->strides[perm[i]];
  }
  for (int i = 0; i < MAX_TENSOR_DIM; i++) {
    layout->shape[i] = new_shape[i];
    layout->strides[i] = new_strides[i];
  }
}

// Returns 1 if the layout is contiguous, 0 otherwise
int cgrad_tensor_layout_is_contiguous(const cgrad_tensor_layout* l) {
  if (!l) return 0;
  uint32_t expected_stride = 1;
  for (int i = MAX_TENSOR_DIM - 1; i >= 0; i--) {
    if (l->strides[i] != expected_stride) return 0;
    expected_stride *= l->shape[i];
  }
  return 1;
}

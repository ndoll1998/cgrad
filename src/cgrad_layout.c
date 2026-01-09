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

// Indexing
size_t cgrad_tensor_flat_index(const uint32_t* indices, const uint32_t* strides) {
  size_t idx = 0;
  for (int i = 0; i < MAX_TENSOR_DIM; i++) {
    idx += indices[i] * strides[i];
  }
  return idx;
}

// Transpose: perm is an array of length MAX_TENSOR_DIM, giving the new order of axes
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

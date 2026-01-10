#include "cgrad_layout.h"
#include "cgrad_errors.h"

/**
 * @brief Deep copy a tensor layout from src to dst.
 * Copies size, shape, and strides.
 * @param dst Destination layout.
 * @param src Source layout.
 */
void cgrad_tensor_layout_copy(cgrad_tensor_layout* dst, const cgrad_tensor_layout* src) {
    if (!dst || !src) return;
    dst->size = src->size;
    for (int i = 0; i < TENSOR_DIM; ++i) {
        dst->shape[i] = src->shape[i];
        dst->strides[i] = src->strides[i];
    }
}

/**
 * @brief Initialize a tensor layout with the given shape and ndim.
 *        The user-specified shape (length ndim) is placed at the end; leading unspecified dims are set to 1.
 *        For example, shape={3,4}, ndim=2, TENSOR_DIM=4 => layout.shape={1,1,3,4}
 */
int cgrad_tensor_layout_init(cgrad_tensor_layout* l, const uint32_t* shape, int ndim) {
  if (!l || !shape) return CGRAD_LAYOUT_ERR_NULL_POINTER;
  if (ndim < 0 || ndim > TENSOR_DIM) return CGRAD_LAYOUT_ERR_SHAPE_MISMATCH;
  uint32_t cur_stride = 1;
  // Fill from the end: user shape at the end, leading dims = 1
  for (int i = TENSOR_DIM - 1; i >= 0; i--) {
    int shape_idx = i - (TENSOR_DIM - ndim);
    uint32_t dim = (shape_idx >= 0) ? shape[shape_idx] : 1;
    l->shape[i] = dim;
  }
  // Compute strides and size
  for (int i = TENSOR_DIM - 1; i >= 0; i--) {
    l->strides[i] = cur_stride;
    cur_stride *= l->shape[i];
  }
  l->size = cur_stride;
  return CGRAD_SUCCESS;
}

/**
 * @brief Compute the flat index in the data array for the given indices and layout.
 *        Indices of length ndim are mapped to the last ndim dims; leading indices behave as 0.
 */
int cgrad_tensor_layout_flat_index(const cgrad_tensor_layout* layout, const uint32_t* indices, int ndim, size_t* out_flat_index) {
  if (!layout || !indices || !out_flat_index) return CGRAD_LAYOUT_ERR_NULL_POINTER;
  if (ndim < 0 || ndim > TENSOR_DIM) return CGRAD_LAYOUT_ERR_SHAPE_MISMATCH;
  size_t idx = 0;
  for (int i = 0; i < TENSOR_DIM; i++) {
    int indices_idx = i - (TENSOR_DIM - ndim);
    uint32_t ind = (indices_idx >= 0) ? indices[indices_idx] : 0;
    if (ind >= layout->shape[i]) {
      return CGRAD_LAYOUT_ERR_INDEX_OUT_OF_BOUNDS;
    }
    idx += ind * layout->strides[i];
  }
  *out_flat_index = idx;
  return CGRAD_SUCCESS;
}

/**
 * @brief Broadcast two layouts between start_dim (inclusive) and end_dim (exclusive).
 * Modifies the layouts in-place to broadcast their shapes and strides.
 * 
 * Broadcasting rules:
 * - If the dimensions at a specific index are the same, do nothing.
 * - If they differ and one is 1, set its stride to 0 and set its shape to the other's shape.
 * - If they differ and neither is 1, return err (cannot broadcast).
 * 
 * After broadcasting, the shapes between the specified dimensions will be the same.
 * @param l1 First layout.
 * @param l2 Second layout.
 * @param start_dim Start dimension (inclusive).
 * @param end_dim End dimension (exclusive).
 * @return 0 on success, -1 on failure.
 */
int cgrad_tensor_layout_broadcast(
    cgrad_tensor_layout* l1,
    cgrad_tensor_layout* l2,
    int start_dim,
    int end_dim
) {
    if (!l1 || !l2) return CGRAD_LAYOUT_ERR_NULL_POINTER;
    if (start_dim < 0 || end_dim > TENSOR_DIM || start_dim >= end_dim) return CGRAD_LAYOUT_ERR_SHAPE_MISMATCH;

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
            return CGRAD_LAYOUT_ERR_BROADCAST;
        }
    }
    return CGRAD_SUCCESS;
}

/**
 * @brief Transpose the layout according to the given permutation, applied to the last ndim dims.
 */
int cgrad_tensor_layout_transpose(cgrad_tensor_layout* layout, const uint32_t* perm, int ndim) {
  if (!layout || !perm) return CGRAD_LAYOUT_ERR_NULL_POINTER;
  if (ndim < 0 || ndim > TENSOR_DIM) return CGRAD_LAYOUT_ERR_SHAPE_MISMATCH;
  // Check for duplicate dimensions in perm
  int seen[TENSOR_DIM] = {0};
  for (int i = 0; i < ndim; i++) {
    if (perm[i] >= ndim) return CGRAD_LAYOUT_ERR_SHAPE_MISMATCH;
    if (seen[perm[i]]) return CGRAD_LAYOUT_ERR_DUPLICATE_DIM;
    seen[perm[i]] = 1;
  }
  // Copy leading dims unchanged, permute last ndim dims
  uint32_t new_shape[TENSOR_DIM];
  uint32_t new_strides[TENSOR_DIM];
  int offset = TENSOR_DIM - ndim;
  for (int i = 0; i < offset; i++) {
    new_shape[i] = layout->shape[i];
    new_strides[i] = layout->strides[i];
  }
  for (int i = 0; i < ndim; i++) {
    new_shape[offset + i] = layout->shape[offset + perm[i]];
    new_strides[offset + i] = layout->strides[offset + perm[i]];
  }
  for (int i = 0; i < TENSOR_DIM; i++) {
    layout->shape[i] = new_shape[i];
    layout->strides[i] = new_strides[i];
  }
  return CGRAD_SUCCESS;
}

/**
 * @brief Returns 1 if the layout is regular (can be traversed with a fixed stride >= 1), 0 otherwise.
 * @param l Pointer to layout.
 * @return 1 if regular, 0 otherwise.
 */
int cgrad_tensor_layout_is_regular(const cgrad_tensor_layout* l) {
  if (!l) return 0;
  uint32_t expected_stride = l->strides[TENSOR_DIM - 1];
  for (int i = TENSOR_DIM - 1; i >= 0; i--) {
    if (l->strides[i] != expected_stride) return 0;
    expected_stride *= l->shape[i];
  }
  return 1;
}

/**
 * @brief Returns 1 if the layout is contiguous, 0 otherwise.
 * @param l Pointer to layout.
 * @return 1 if contiguous, 0 otherwise.
 */
int cgrad_tensor_layout_is_contiguous(const cgrad_tensor_layout* l) {
  return cgrad_tensor_layout_is_regular(l) && (l->strides[TENSOR_DIM - 1] == 1);
}
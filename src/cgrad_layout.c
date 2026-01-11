#include "cgrad_layout.h"
#include "cgrad_errors.h"
#include <string.h>

/**
 * @brief Deep copy a tensor layout from src to dst.
 * Copies size, shape, and strides.
 * @param dst Destination layout.
 * @param src Source layout.
 */
void cgrad_tensor_layout_copy(cgrad_tensor_layout* dst, const cgrad_tensor_layout* src) {
    if (!dst || !src) return;
    memcpy(dst, src, sizeof(cgrad_tensor_layout));
}

/**
 * @brief Initialize a tensor layout with the given shape and ndim.
 *        The user-specified shape (length ndim) is placed at the end; leading unspecified dims are set to 1.
 *        For example, shape={3,4}, ndim=2, TENSOR_DIM=4 => layout.shape={1,1,3,4}
 */
int cgrad_tensor_layout_init(cgrad_tensor_layout* l, const uint32_t* shape, int ndim) {
  if (!l || !shape) return CGRAD_LAYOUT_ERR_NULL_POINTER;
  if (ndim < 0 || ndim > TENSOR_DIM) return CGRAD_LAYOUT_ERR_SHAPE_MISMATCH;
  memset(l, 0, sizeof(*l));
  // Fill from the end: user shape at the end, leading dims = 1
  for (int i = 0; i < TENSOR_DIM; i++) {
    int shape_idx = i - (TENSOR_DIM - ndim);
    uint32_t dim = (shape_idx >= 0) ? shape[shape_idx] : 1;
    l->shape[i] = dim;
  }
  // Compute strides
  l->strides[TENSOR_DIM - 1] = 1;
  for (int i = TENSOR_DIM - 2; i >= 0; i--) {
    l->strides[i] = l->strides[i + 1] * l->shape[i + 1];
  }
  // Compute size
  uint32_t size = 1;
  for (int i = 0; i < TENSOR_DIM; i++) {
    size *= l->shape[i];
  }
  l->size = size;
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
    uint32_t ind = 0;
    if (i >= TENSOR_DIM - ndim) {
      ind = indices[i - (TENSOR_DIM - ndim)];
    }
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
  // Overwrite shape and strides in layout
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
  if (TENSOR_DIM == 0) return 1;
  // Find the scaling factor k (stride of last dim)
  uint32_t k = l->strides[TENSOR_DIM - 1];
  if (k == 0) return 0;
  // Check all strides
  uint32_t expected = k;
  for (int i = TENSOR_DIM - 1; i >= 0; i--) {
    if (l->strides[i] != expected) return 0;
    if (i > 0) expected *= l->shape[i];
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

/**
 * @brief Reshape the layout to a new shape (with at most one -1 to infer dimension).
 *        The layout must be regular. Updates shape and strides in-place.
 *        The new strides are computed as for a regular layout, but scaled by the original step size (last stride).
 *        Returns error if the shape is invalid or the layout is not regular.
 * @param layout Pointer to layout to reshape.
 * @param new_shape Array of new dimensions (length ndim, may contain one -1).
 * @param ndim Number of dimensions in new_shape (<= TENSOR_DIM).
 * @return CGRAD_SUCCESS on success,
 *         CGRAD_LAYOUT_ERR_RESHAPE_INVALID_SHAPE if shape is invalid,
 *         CGRAD_LAYOUT_ERR_NOT_REGULAR if layout is not regular.
 */
int cgrad_tensor_layout_reshape(cgrad_tensor_layout* layout, const int32_t* new_shape, int ndim) {
  if (!layout || !new_shape) return CGRAD_LAYOUT_ERR_NULL_POINTER;
  if (ndim < 0 || ndim > TENSOR_DIM) return CGRAD_LAYOUT_ERR_SHAPE_MISMATCH;
  if (!cgrad_tensor_layout_is_regular(layout)) return CGRAD_LAYOUT_ERR_NOT_REGULAR;

  // Compute total elements in old layout
  uint32_t old_size = 1;
  for (int i = 0; i < TENSOR_DIM; ++i) {
    old_size *= layout->shape[i];
  }

  // Validate new_shape and find -1
  int minus1_idx = -1;
  uint32_t new_size = 1;
  for (int i = 0; i < ndim; ++i) {
    if (new_shape[i] == -1) {
      if (minus1_idx != -1) return CGRAD_LAYOUT_ERR_RESHAPE_INVALID_SHAPE; // More than one -1
      minus1_idx = i;
    } else if (new_shape[i] <= 0) {
      return CGRAD_LAYOUT_ERR_RESHAPE_INVALID_SHAPE;
    } else {
      new_size *= (uint32_t)new_shape[i];
    }
  }

  uint32_t inferred_dim = 0;
  if (minus1_idx != -1) {
    if (new_size == 0 || old_size % new_size != 0) return CGRAD_LAYOUT_ERR_RESHAPE_INVALID_SHAPE;
    inferred_dim = old_size / new_size;
    if (inferred_dim == 0) return CGRAD_LAYOUT_ERR_RESHAPE_INVALID_SHAPE;
  } else {
    if (new_size != old_size) return CGRAD_LAYOUT_ERR_RESHAPE_INVALID_SHAPE;
  }

  // Fill new shape into layout->shape (right-aligned, leading dims set to 1)
  int offset = TENSOR_DIM - ndim;
  for (int i = 0; i < TENSOR_DIM; ++i) {
    uint32_t dim;
    if (i < offset) {
      dim = 1;
    } else if ((i - offset) == minus1_idx) {
      dim = (minus1_idx != -1) ? inferred_dim : (uint32_t)new_shape[i - offset];
    } else {
      dim = (uint32_t)new_shape[i - offset];
    }
    layout->shape[i] = dim;
  }

  // Update size
  uint32_t final_size = 1;
  for (int i = 0; i < TENSOR_DIM; ++i) {
    final_size *= layout->shape[i];
  }
  layout->size = final_size;

  // Compute new strides: like contiguous, but scale by original step size
  uint32_t step = layout->strides[TENSOR_DIM - 1];
  uint32_t cur_stride = step;
  for (int i = TENSOR_DIM - 1; i >= 0; --i) {
    layout->strides[i] = cur_stride;
    cur_stride *= layout->shape[i];
  }

  return CGRAD_SUCCESS;
}

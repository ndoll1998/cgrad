#ifndef CGRAD_STORAGE_LAYOUT_H
#define CGRAD_STORAGE_LAYOUT_H

#include <stdint.h>
#include <stddef.h>
#include "cgrad_status.h"

#define TENSOR_DIM 8

/**
 * @brief Structure representing the layout (shape, strides, size) of a tensor.
 */
typedef struct cgrad_storage_layout {
  uint32_t size;                  /**< Total number of elements */
  uint32_t shape[TENSOR_DIM]; /**< Shape of each dimension */
  uint32_t strides[TENSOR_DIM]; /**< Strides for each dimension */
} cgrad_storage_layout;

// --- Copy/Initialization ---

/**
 * @brief Deep copy a tensor layout from src to dst.
 * Copies size, shape, and strides.
 * @param dst Destination layout.
 * @param src Source layout.
 */
void cgrad_storage_layout_copy(cgrad_storage_layout* dst, const cgrad_storage_layout* src);

/**
 * @brief Initialize a tensor layout with the given shape and ndim.
 *        The user-specified shape (length ndim) is placed at the end; leading unspecified dims are set to 1.
 *        For example, shape={3,4}, ndim=2, TENSOR_DIM=4 => layout.shape={1,1,3,4}
 * @param l Pointer to layout to initialize.
 * @param shape Array of dimensions (length ndim).
 * @param ndim Number of dimensions in shape (<= TENSOR_DIM).
 * @return 0 on success, error code otherwise.
 */
cgrad_status cgrad_storage_layout_init(cgrad_storage_layout* l, const uint32_t* shape, int ndim);

// --- Indexing ---

/**
 * @brief Compute the flat index in the data array for the given indices and layout.
 *        Checks that all indices are within bounds (0 <= idx < shape[i]).
 *        Indices of length ndim are mapped to the last ndim dims; leading indices behave as 0.
 *        For example, indices={2,3}, ndim=2, TENSOR_DIM=4 => layout.shape={1,1,3,4}, indices used as {0,0,2,3}
 *        If any index is out of bounds, returns CGRAD_ERR_STORAGE_LAYOUT_INDEX_OUT_OF_BOUNDS.
 * @param layout Pointer to tensor layout (provides shape and strides).
 * @param indices Array of indices (length ndim).
 * @param ndim Number of dimensions in indices (<= TENSOR_DIM).
 * @param out_flat_index Pointer to size_t where the computed flat index will be stored.
 * @return CGRAD_SUCCESS on success, CGRAD_ERR_STORAGE_LAYOUT_INDEX_OUT_OF_BOUNDS if any index is out of bounds.
 */
cgrad_status cgrad_storage_layout_flat_index(const cgrad_storage_layout* layout, const uint32_t* indices, int ndim, size_t* out_flat_index);

// --- Broadcasting ---

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
cgrad_status cgrad_storage_layout_broadcast(
    cgrad_storage_layout* l1,
    cgrad_storage_layout* l2,
    int start_dim,
    int end_dim
);

// --- Transform ---

/**
 * @brief Transpose the layout according to the given permutation, applied to the last ndim dims.
 *        Returns an error if any dimension is repeated in perm.
 *        For example, perm={1,0}, ndim=2, TENSOR_DIM=4 => perm applied to dims 2 and 3.
 * @param layout Pointer to layout.
 * @param perm Permutation array (length ndim).
 * @param ndim Number of dimensions to permute (<= TENSOR_DIM).
 * @return 0 on success, CGRAD_ERR_STORAGE_LAYOUT_DUPLICATE_DIM if a dimension is repeated.
 */
cgrad_status cgrad_storage_layout_transpose(cgrad_storage_layout* layout, const uint32_t* perm, int ndim);

// --- Info ---

/**
 * @brief Returns 1 if the layout is regular (can be traversed with a fixed stride >= 1), 0 otherwise.
 * @param l Pointer to layout.
 * @return 1 if regular, 0 otherwise.
 */
int cgrad_storage_layout_is_regular(const cgrad_storage_layout* l);

/**
 * @brief Returns 1 if the layout is contiguous, 0 otherwise.
 * @param l Pointer to layout.
 * @return 1 if contiguous, 0 otherwise.
 */
int cgrad_storage_layout_is_contiguous(const cgrad_storage_layout* l);

/**
 * @brief Reshape the layout to a new shape (with at most one -1 to infer dimension).
 *        The layout must be regular. Updates shape and strides in-place.
 *        The new strides are computed as for a regular layout, but scaled by the original step size (last stride).
 *        Returns error if the shape is invalid or the layout is not regular.
 * @param layout Pointer to layout to reshape.
 * @param new_shape Array of new dimensions (length ndim, may contain one -1).
 * @param ndim Number of dimensions in new_shape (<= TENSOR_DIM).
 * @return CGRAD_SUCCESS on success,
 *         CGRAD_ERR_STORAGE_LAYOUT_RESHAPE_INVALID_SHAPE if shape is invalid,
 *         CGRAD_ERR_STORAGE_LAYOUT_NOT_REGULAR if layout is not regular.
 */
cgrad_status cgrad_storage_layout_reshape(cgrad_storage_layout* layout, const int32_t* new_shape, int ndim);

/**
 * @brief Print the shape of the layout in the format (d0, d1, ..., dn).
 * @param l Pointer to layout.
 * @param ndim Number of dimensions to print (<= TENSOR_DIM).
 */
void cgrad_storage_layout_print_shape(const cgrad_storage_layout* l, int ndim);

/**
 * @brief Apply a reduction mask to a layout, computing the output shape.
 *        Dimensions where mask is 1 are reduced to size 1.
 *        Updates shape, strides, and size in-place.
 * 
 * The mask is applied to the last ndim dimensions of the layout.
 * For example, with mask={1,0} and ndim=2 on a (2,3,4,5) tensor:
 * - Last 2 dims are (4,5)
 * - Apply mask: dim -2 becomes 1, dim -1 stays 5
 * - Result shape: (2,3,1,5)
 * 
 * @param layout Pointer to layout to reduce.
 * @param mask Reduction mask array (length ndim, 1=reduce, 0=keep).
 * @param ndim Number of dimensions in mask (<= TENSOR_DIM).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
cgrad_status cgrad_storage_layout_reduce(cgrad_storage_layout* layout, const uint8_t* mask, int ndim);

#endif // CGRAD_STORAGE_LAYOUT_H

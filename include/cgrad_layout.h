#ifndef CGRAD_LAYOUT_H
#define CGRAD_LAYOUT_H

#include <stdint.h>
#include <stddef.h>

#define MAX_TENSOR_DIM 4

/**
 * @brief Structure representing the layout (shape, strides, size) of a tensor.
 */
typedef struct cgrad_tensor_layout {
  uint32_t size;                  /**< Total number of elements */
  uint32_t shape[MAX_TENSOR_DIM]; /**< Shape of each dimension */
  uint32_t strides[MAX_TENSOR_DIM]; /**< Strides for each dimension */
} cgrad_tensor_layout;

// --- Copy/Initialization ---

/**
 * @brief Deep copy a tensor layout from src to dst.
 * Copies size, shape, and strides.
 * @param dst Destination layout.
 * @param src Source layout.
 */
void cgrad_tensor_layout_copy(cgrad_tensor_layout* dst, const cgrad_tensor_layout* src);

/**
 * @brief Initialize a tensor layout with the given shape and ndim.
 *        The user-specified shape (length ndim) is placed at the end; leading unspecified dims are set to 1.
 *        For example, shape={3,4}, ndim=2, MAX_TENSOR_DIM=4 => layout.shape={1,1,3,4}
 * @param l Pointer to layout to initialize.
 * @param shape Array of dimensions (length ndim).
 * @param ndim Number of dimensions in shape (<= MAX_TENSOR_DIM).
 * @return 0 on success, error code otherwise.
 */
int cgrad_tensor_layout_init(cgrad_tensor_layout* l, const uint32_t* shape, int ndim);

// --- Indexing ---

/**
 * @brief Compute the flat index in the data array for the given indices and layout.
 *        Checks that all indices are within bounds (0 <= idx < shape[i]).
 *        Indices of length ndim are mapped to the last ndim dims; leading indices behave as 0.
 *        For example, indices={2,3}, ndim=2, MAX_TENSOR_DIM=4 => layout.shape={1,1,3,4}, indices used as {0,0,2,3}
 *        If any index is out of bounds, returns CGRAD_LAYOUT_ERR_INDEX_OUT_OF_BOUNDS.
 * @param layout Pointer to tensor layout (provides shape and strides).
 * @param indices Array of indices (length ndim).
 * @param ndim Number of dimensions in indices (<= MAX_TENSOR_DIM).
 * @param out_flat_index Pointer to size_t where the computed flat index will be stored.
 * @return CGRAD_SUCCESS on success, CGRAD_LAYOUT_ERR_INDEX_OUT_OF_BOUNDS if any index is out of bounds.
 */
int cgrad_tensor_layout_flat_index(const cgrad_tensor_layout* layout, const uint32_t* indices, int ndim, size_t* out_flat_index);

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
int cgrad_tensor_layout_broadcast(
    cgrad_tensor_layout* l1,
    cgrad_tensor_layout* l2,
    int start_dim,
    int end_dim
);

// --- Transform ---

/**
 * @brief Transpose the layout according to the given permutation, applied to the last ndim dims.
 *        Returns an error if any dimension is repeated in perm.
 *        For example, perm={1,0}, ndim=2, MAX_TENSOR_DIM=4 => perm applied to dims 2 and 3.
 * @param layout Pointer to layout.
 * @param perm Permutation array (length ndim).
 * @param ndim Number of dimensions to permute (<= MAX_TENSOR_DIM).
 * @return 0 on success, CGRAD_LAYOUT_ERR_DUPLICATE_DIM if a dimension is repeated.
 */
int cgrad_tensor_layout_transpose(cgrad_tensor_layout* layout, const uint32_t* perm, int ndim);

// --- Info ---

/**
 * @brief Returns 1 if the layout is regular (can be traversed with a fixed stride >= 1), 0 otherwise.
 * @param l Pointer to layout.
 * @return 1 if regular, 0 otherwise.
 */
int cgrad_tensor_layout_is_regular(const cgrad_tensor_layout* l);

/**
 * @brief Returns 1 if the layout is contiguous, 0 otherwise.
 * @param l Pointer to layout.
 * @return 1 if contiguous, 0 otherwise.
 */
int cgrad_tensor_layout_is_contiguous(const cgrad_tensor_layout* l);

#endif // CGRAD_LAYOUT_H
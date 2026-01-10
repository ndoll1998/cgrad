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
 * @brief Initialize a tensor layout with the given shape.
 * @param l Pointer to layout to initialize.
 * @param shape Array of dimensions.
 * @return 0 on success, error code otherwise.
 */
int cgrad_tensor_layout_init(cgrad_tensor_layout* l, const uint32_t* shape);

// --- Indexing ---

/**
 * @brief Compute the flat index in the data array for the given indices and strides.
 * @param indices Array of indices.
 * @param strides Array of strides.
 * @return Flat index.
 */
size_t cgrad_tensor_flat_index(const uint32_t* indices, const uint32_t* strides);

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
 * @brief Transpose the layout according to the given permutation.
 *        Returns an error if any dimension is repeated in perm.
 * @param layout Pointer to layout.
 * @param perm Permutation array.
 * @return 0 on success, CGRAD_LAYOUT_ERR_DUPLICATE_DIM if a dimension is repeated.
 */
int cgrad_tensor_layout_transpose(cgrad_tensor_layout* layout, const uint32_t* perm);

// --- Info ---

/**
 * @brief Returns 1 if the layout is contiguous, 0 otherwise.
 * @param l Pointer to layout.
 * @return 1 if contiguous, 0 otherwise.
 */
int cgrad_tensor_layout_is_contiguous(const cgrad_tensor_layout* l);

#endif // CGRAD_LAYOUT_H
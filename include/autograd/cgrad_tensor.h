#ifndef CGRAD_TENSOR_H
#define CGRAD_TENSOR_H

#include <uuid/uuid.h>
#include "autograd/cgrad_compute_graph.h"
#include "storage/cgrad_storage_layout.h"
#include "storage/cgrad_storage_backend.h"
#include "storage/cgrad_storage_registry.h"

/**
 * @file cgrad_tensor.h
 * @brief Lazy tensor API for cgrad computation graphs.
 * 
 * A tensor is a reference to a computation graph node. For leaf nodes (inputs),
 * materialized eager storage is managed internally via the graph node's cached_storage.
 * For operation nodes, the tensor represents a deferred computation.
 * 
 * Users do not interact with cgrad_storage directly; all operations go through
 * the tensor interface which manages storage internally.
 */

/**
 * @brief A tensor in the computation graph.
 * 
 * A tensor represents a value in the computation graph. For leaf nodes (inputs),
 * materialized eager storage is managed internally via the graph node's cached_storage.
 * For operation nodes, the tensor represents a deferred computation.
 * 
 * All tensors reference a single global computation graph, simplifying graph management
 * and eliminating the need for graph merging.
 * 
 * Users do not interact with cgrad_storage directly; all operations go through
 * the tensor interface which manages storage internally.
 */
typedef struct {
    uuid_t node_id;                  /**< ID of the node producing this tensor */
    cgrad_storage_layout layout;     /**< Shape and layout of the tensor */
} cgrad_tensor;

// ============================================================================
// Tensor Initialization and Management
// ============================================================================

/**
 * @brief Initialize an input tensor with the given shape.
 * 
 * Creates a new computation graph with a single leaf node.
 * The underlying eager storage is managed internally by the tensor.
 * Use cgrad_tensor_fill_* functions to initialize the data.
 * 
 * @param tensor Pointer to tensor to initialize.
 * @param shape Array of dimensions (length ndim).
 * @param ndim Number of dimensions (≤ TENSOR_DIM).
 * @param backend_type Which backend to use for storage.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_init(
    cgrad_tensor* tensor,
    const uint32_t* shape,
    int ndim,
    cgrad_storage_backend_type backend_type
);

/**
 * @brief Fill a tensor with a constant value.
 * 
 * @param tensor Tensor to fill.
 * @param value The value to fill with.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_fill(cgrad_tensor* tensor, float value);

/**
 * @brief Fill a tensor with random values.
 * 
 * @param tensor Tensor to fill with random data.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_fill_rand(cgrad_tensor* tensor);

/**
 * @brief Free a tensor using reference counting.
 * 
 * Decrements the reference count of the node. If the reference count reaches zero,
 * the node and its storage are freed, and input nodes have their reference counts
 * decremented recursively.
 * 
 * @param tensor Tensor to free.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_free(cgrad_tensor* tensor);

/**
 * @brief Copy a tensor (increments reference count of the referenced node).
 * 
 * Creates a new tensor that references the same computation graph node.
 * The reference count of the node is incremented.
 * 
 * @param src Source tensor.
 * @param dst Destination tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_copy(const cgrad_tensor* src, cgrad_tensor* dst);

/**
 * @brief Cleanup the global compute graph.
 * 
 * This should be called at program shutdown to free all graph resources.
 * After calling this, no tensor operations should be performed.
 */
void cgrad_tensor_cleanup_global_graph(void);

// ============================================================================
// Binary Operations
// ============================================================================

/**
 * @brief Element-wise addition of two tensors.
 * 
 * If the tensors reference different graphs, they are merged.
 * Performs shape broadcasting as needed.
 * 
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param out_tensor Pointer to output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_add(
    const cgrad_tensor* a,
    const cgrad_tensor* b,
    cgrad_tensor* out_tensor
);

/**
 * @brief Element-wise subtraction: out = a - b
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param out_tensor Pointer to output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_sub(
    const cgrad_tensor* a,
    const cgrad_tensor* b,
    cgrad_tensor* out_tensor
);

/**
 * @brief Batched matrix multiplication: out = a @ b
 * @param a First input tensor (shape: ..., m, k).
 * @param b Second input tensor (shape: ..., k, n).
 * @param out_tensor Pointer to output tensor (shape: ..., m, n).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_gemm(
    const cgrad_tensor* a,
    const cgrad_tensor* b,
    cgrad_tensor* out_tensor
);

// ============================================================================
// Unary Operations
// ============================================================================

/**
 * @brief Transpose tensor along specified axes.
 * 
 * The permutation is applied to the last ndim dimensions.
 * For example, with a (2,3,4,5) tensor and perm={1,0}, ndim=2:
 * - Axes 2,3 are permuted to get shape (2,3,5,4)
 * 
 * @param tensor Input tensor.
 * @param perm Permutation array (length ndim).
 * @param ndim Number of dimensions to permute.
 * @param out_tensor Pointer to output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_transpose(
    const cgrad_tensor* tensor,
    const uint32_t* perm,
    int ndim,
    cgrad_tensor* out_tensor
);

/**
 * @brief Reshape tensor to new dimensions.
 * 
 * The new shape must be compatible with the current shape
 * (same total number of elements).
 * One dimension may be -1 to infer its size.
 * 
 * @param tensor Input tensor.
 * @param new_shape Array of new dimensions.
 * @param ndim Number of dimensions in new_shape.
 * @param out_tensor Pointer to output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_reshape(
    const cgrad_tensor* tensor,
    const int32_t* new_shape,
    int ndim,
    cgrad_tensor* out_tensor
);

/**
 * @brief Sum reduction along specified axes.
 * 
 * The mask indicates which axes to reduce (1=sum, 0=keep).
 * Example: mask={1,0} on shape (2,3,4) -> shape (1,3,1)
 * 
 * @param tensor Input tensor.
 * @param mask Reduction mask (length ndim).
 * @param ndim Number of dimensions.
 * @param out_tensor Pointer to output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_reduce_sum(
    const cgrad_tensor* tensor,
    const uint8_t* mask,
    int ndim,
    cgrad_tensor* out_tensor
);

// ============================================================================
// Execution
// ============================================================================

/**
 * @brief Execute the computation graph and materialize a tensor.
 * 
 * This function traces dependencies from the target node back to cached nodes or leaves,
 * computes required intermediate results, and caches them in the tensor's node.
 * 
 * Execution strategy:
 * 1. Reverse topological traversal from target node to identify dependencies
 * 2. Stop traversal at nodes that already have cached storage
 * 3. Execute non-cached nodes from their dependencies (either cached nodes or leaves)
 * 4. Cache results in each node's storage field for reuse in future executions
 * 
 * This avoids redundant computation when multiple tensors share subgraphs or
 * when a tensor is executed multiple times.
 * 
 * @param tensor Tensor to materialize (specifies which node to compute).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_execute(cgrad_tensor* tensor);

/**
 * @brief Get the underlying storage of an executed tensor.
 * 
 * This function returns the cached storage after execution.
 * Returns NULL if the tensor has not been executed yet.
 * 
 * @param tensor Tensor to get storage from.
 * @return Pointer to storage, or NULL if not executed.
 */
cgrad_storage* cgrad_tensor_get_storage(const cgrad_tensor* tensor);

/**
 * @brief Print tensor information.
 * @param tensor Tensor to print.
 */
void cgrad_tensor_print(const cgrad_tensor* tensor);

// ============================================================================
// Gradient Functions
// ============================================================================

/**
 * @brief Set whether a tensor requires gradient computation.
 * 
 * This should be called on leaf tensors (inputs) before building the computation graph.
 * Operation tensors automatically inherit requires_grad from their inputs.
 * 
 * @param tensor Tensor to configure.
 * @param requires_grad 1 to enable gradient computation, 0 to disable.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_set_requires_grad(cgrad_tensor* tensor, int requires_grad);

/**
 * @brief Check if a tensor requires gradient computation.
 * 
 * @param tensor Tensor to query.
 * @param out_requires_grad Pointer to output flag.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_get_requires_grad(const cgrad_tensor* tensor, int* out_requires_grad);

/**
 * @brief Get the gradient storage of a tensor after backward pass.
 * 
 * Returns NULL if backward has not been called or if the tensor doesn't require gradients.
 * 
 * @param tensor Tensor to get gradient from.
 * @return Pointer to gradient storage, or NULL if not available.
 */
cgrad_storage* cgrad_tensor_get_grad(const cgrad_tensor* tensor);

/**
 * @brief Zero out all gradients in the computation graph.
 * 
 * This should be called before each backward pass in training loops.
 * 
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_zero_grad(void);

/**
 * @brief Compute gradients by backpropagation through the computation graph.
 * 
 * This function:
 * 1. Initializes the gradient of the target tensor to 1.0 (or provided grad_output)
 * 2. Traverses the graph in reverse topological order
 * 3. For each node with requires_grad=1, computes gradients of inputs using the backward function
 * 4. Accumulates gradients when a node is used multiple times
 * 5. Stores gradients in each node's grad_storage field
 * 
 * The target tensor must have been executed (forward pass) before calling backward.
 * 
 * @param tensor Target tensor (typically a scalar loss).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_backward(cgrad_tensor* tensor);

#endif // CGRAD_TENSOR_H

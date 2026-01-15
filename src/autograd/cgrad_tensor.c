#include "autograd/cgrad_tensor.h"
#include "cgrad_errors.h"
#include "storage/cgrad_storage.h"
#include "storage/cgrad_storage_registry.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Global Compute Graph
// ============================================================================

static cgrad_compute_graph* g_global_graph = NULL;

/**
 * @brief Get or create the global compute graph.
 */
static cgrad_compute_graph* get_global_graph(void) {
    if (g_global_graph == NULL) {
        g_global_graph = (cgrad_compute_graph*)malloc(sizeof(cgrad_compute_graph));
        if (g_global_graph == NULL) {
            return NULL;
        }
        
        int ret = cgrad_compute_graph_create(g_global_graph);
        if (ret != CGRAD_SUCCESS) {
            free(g_global_graph);
            g_global_graph = NULL;
            return NULL;
        }
    }
    
    return g_global_graph;
}

/**
 * @brief Free the global compute graph.
 * This should be called at program shutdown.
 */
void cgrad_tensor_cleanup_global_graph(void) {
    if (g_global_graph != NULL) {
        cgrad_compute_graph_free(g_global_graph);
        free(g_global_graph);
        g_global_graph = NULL;
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Infer output shape for binary operations with broadcasting.
 */
static int infer_binary_output_shape(const cgrad_storage_layout* a_layout,
                                     const cgrad_storage_layout* b_layout,
                                     cgrad_storage_layout* out_layout) {
    // Copy layouts for broadcasting (broadcast modifies in-place)
    cgrad_storage_layout result_a = *a_layout;
    cgrad_storage_layout result_b = *b_layout;
    int ret = cgrad_storage_layout_broadcast(&result_a, &result_b, 0, TENSOR_DIM);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }
    
    *out_layout = result_a;  // Both result_a and result_b have the same shape
    return CGRAD_SUCCESS;
}

/**
 * @brief Infer output shape for GEMM operation.
 */
static int infer_gemm_output_shape(const cgrad_storage_layout* a_layout,
                                   const cgrad_storage_layout* b_layout,
                                   cgrad_storage_layout* out_layout) {
    // For GEMM: a is (..., m, k), b is (..., k, n), output is (..., m, n)
    // Work with last 2 dimensions of TENSOR_DIM
    int m = a_layout->shape[TENSOR_DIM - 2];
    int k_a = a_layout->shape[TENSOR_DIM - 1];
    int k_b = b_layout->shape[TENSOR_DIM - 2];
    int n = b_layout->shape[TENSOR_DIM - 1];

    if (k_a != k_b) {
        return CGRAD_GRAPH_ERR_SHAPE_MISMATCH;
    }

    // Copy entire layout from a
    *out_layout = *a_layout;
    // Just modify the last dimension to n
    out_layout->shape[TENSOR_DIM - 1] = n;
    
    // Recalculate strides based on new shape
    out_layout->strides[TENSOR_DIM - 1] = 1;
    for (int i = TENSOR_DIM - 2; i >= 0; i--) {
        out_layout->strides[i] = out_layout->strides[i + 1] * out_layout->shape[i + 1];
    }
    
    // Recalculate size
    out_layout->size = 1;
    for (int i = 0; i < TENSOR_DIM; i++) {
        out_layout->size *= out_layout->shape[i];
    }
    
    return CGRAD_SUCCESS;
}

// ============================================================================
// Tensor Initialization and Management
// ============================================================================

int cgrad_tensor_init(
    cgrad_tensor* tensor,
    const uint32_t* shape,
    int ndim,
    cgrad_storage_backend_type backend_type
) {
    if (tensor == NULL || shape == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    // Get global graph
    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Initialize layout
    int ret = cgrad_storage_layout_init(&tensor->layout, shape, ndim);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    // Create storage
    cgrad_storage* storage = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    if (storage == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    ret = cgrad_storage_init(storage, shape, ndim, backend_type);
    if (ret != CGRAD_SUCCESS) {
        free(storage);
        return ret;
    }

    // Add leaf node to global graph
    ret = cgrad_compute_graph_add_leaf(graph, &tensor->layout, storage, tensor->node_id);
    if (ret != CGRAD_SUCCESS) {
        cgrad_storage_free(storage);
        free(storage);
        return ret;
    }

    return CGRAD_SUCCESS;
}

int cgrad_tensor_fill(cgrad_tensor* tensor, float value) {
    if (tensor == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Get the node
    cgrad_graph_node* node;
    int ret = cgrad_compute_graph_get_node(graph, tensor->node_id, &node);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    // Fill the storage
    if (node->storage == NULL) {
        return CGRAD_GRAPH_ERR_EXECUTION_FAILED;
    }

    return cgrad_storage_fill(node->storage, value);
}

int cgrad_tensor_fill_rand(cgrad_tensor* tensor) {
    if (tensor == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Get the node
    cgrad_graph_node* node;
    int ret = cgrad_compute_graph_get_node(graph, tensor->node_id, &node);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    // Fill the storage
    if (node->storage == NULL) {
        return CGRAD_GRAPH_ERR_EXECUTION_FAILED;
    }

    return cgrad_storage_fill_rand(node->storage);
}

int cgrad_tensor_free(cgrad_tensor* tensor) {
    if (tensor == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Delegate to compute graph layer for reference counting
    return cgrad_compute_graph_decrement_ref(graph, tensor->node_id);
}

int cgrad_tensor_copy(const cgrad_tensor* src, cgrad_tensor* dst) {
    if (src == NULL || dst == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Copy tensor structure
    uuid_copy(dst->node_id, src->node_id);
    dst->layout = src->layout;

    // Delegate to compute graph layer for reference counting
    return cgrad_compute_graph_increment_ref(graph, src->node_id);
}

// ============================================================================
// Binary Operations
// ============================================================================

int cgrad_tensor_add(
    const cgrad_tensor* a,
    const cgrad_tensor* b,
    cgrad_tensor* out_tensor
) {
    if (a == NULL || b == NULL || out_tensor == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Determine output shape
    cgrad_storage_layout out_layout;
    int ret = infer_binary_output_shape(&a->layout, &b->layout, &out_layout);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    // Create operation node
    cgrad_op_info op_info;
    op_info.type = CGRAD_OP_ADD;

    uuid_t input_ids[2];
    uuid_copy(input_ids[0], a->node_id);
    uuid_copy(input_ids[1], b->node_id);

    ret = cgrad_compute_graph_add_op(
        graph, &op_info, &out_layout,
        input_ids, 2, out_tensor->node_id
    );
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    out_tensor->layout = out_layout;
    return CGRAD_SUCCESS;
}

int cgrad_tensor_sub(
    const cgrad_tensor* a,
    const cgrad_tensor* b,
    cgrad_tensor* out_tensor
) {
    if (a == NULL || b == NULL || out_tensor == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Determine output shape
    cgrad_storage_layout out_layout;
    int ret = infer_binary_output_shape(&a->layout, &b->layout, &out_layout);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    // Create operation node
    cgrad_op_info op_info;
    op_info.type = CGRAD_OP_SUB;

    uuid_t input_ids[2];
    uuid_copy(input_ids[0], a->node_id);
    uuid_copy(input_ids[1], b->node_id);

    ret = cgrad_compute_graph_add_op(
        graph, &op_info, &out_layout,
        input_ids, 2, out_tensor->node_id
    );
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    out_tensor->layout = out_layout;
    return CGRAD_SUCCESS;
}

int cgrad_tensor_gemm(
    const cgrad_tensor* a,
    const cgrad_tensor* b,
    cgrad_tensor* out_tensor
) {
    if (a == NULL || b == NULL || out_tensor == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Determine output shape
    cgrad_storage_layout out_layout;
    int ret = infer_gemm_output_shape(&a->layout, &b->layout, &out_layout);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    // Create operation node
    cgrad_op_info op_info;
    op_info.type = CGRAD_OP_GEMM;

    uuid_t input_ids[2];
    uuid_copy(input_ids[0], a->node_id);
    uuid_copy(input_ids[1], b->node_id);

    ret = cgrad_compute_graph_add_op(
        graph, &op_info, &out_layout,
        input_ids, 2, out_tensor->node_id
    );
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    out_tensor->layout = out_layout;
    return CGRAD_SUCCESS;
}

// ============================================================================
// Unary Operations
// ============================================================================

int cgrad_tensor_transpose(
    const cgrad_tensor* tensor,
    const uint32_t* perm,
    int ndim,
    cgrad_tensor* out_tensor
) {
    if (tensor == NULL || perm == NULL || out_tensor == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Compute output layout
    cgrad_storage_layout out_layout = tensor->layout;
    int ret = cgrad_storage_layout_transpose(&out_layout, perm, ndim);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    // Create operation node
    cgrad_op_info op_info;
    op_info.type = CGRAD_OP_TRANSPOSE;
    op_info.metadata.transpose.ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        op_info.metadata.transpose.perm[i] = perm[i];
    }

    uuid_t input_ids[1];
    uuid_copy(input_ids[0], tensor->node_id);

    ret = cgrad_compute_graph_add_op(
        graph, &op_info, &out_layout,
        input_ids, 1, out_tensor->node_id
    );
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    out_tensor->layout = out_layout;
    return CGRAD_SUCCESS;
}

int cgrad_tensor_reshape(
    const cgrad_tensor* tensor,
    const int32_t* new_shape,
    int ndim,
    cgrad_tensor* out_tensor
) {
    if (tensor == NULL || new_shape == NULL || out_tensor == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Compute output layout
    cgrad_storage_layout out_layout = tensor->layout;
    int ret = cgrad_storage_layout_reshape(&out_layout, new_shape, ndim);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    // Create operation node
    cgrad_op_info op_info;
    op_info.type = CGRAD_OP_RESHAPE;
    op_info.metadata.reshape.ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        op_info.metadata.reshape.new_shape[i] = new_shape[i];
    }

    uuid_t input_ids[1];
    uuid_copy(input_ids[0], tensor->node_id);

    ret = cgrad_compute_graph_add_op(
        graph, &op_info, &out_layout,
        input_ids, 1, out_tensor->node_id
    );
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    out_tensor->layout = out_layout;
    return CGRAD_SUCCESS;
}

int cgrad_tensor_reduce_sum(
    const cgrad_tensor* tensor,
    const uint8_t* mask,
    int ndim,
    cgrad_tensor* out_tensor
) {
    if (tensor == NULL || mask == NULL || out_tensor == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Compute output layout using the layout reduce function
    cgrad_storage_layout out_layout = tensor->layout;
    int ret = cgrad_storage_layout_reduce(&out_layout, mask, ndim);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    // Create operation node
    cgrad_op_info op_info;
    op_info.type = CGRAD_OP_REDUCE_SUM;
    op_info.metadata.reduce_sum.ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        op_info.metadata.reduce_sum.mask[i] = mask[i];
    }

    uuid_t input_ids[1];
    uuid_copy(input_ids[0], tensor->node_id);

    ret = cgrad_compute_graph_add_op(
        graph, &op_info, &out_layout,
        input_ids, 1, out_tensor->node_id
    );
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    out_tensor->layout = out_layout;
    return CGRAD_SUCCESS;
}

// ============================================================================
// Execution
// ============================================================================

int cgrad_tensor_execute(cgrad_tensor* tensor) {
    if (!tensor) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (!graph) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Execute the subgraph for this tensor
    return cgrad_compute_graph_execute(graph, tensor->node_id);
}

cgrad_storage* cgrad_tensor_get_storage(const cgrad_tensor* tensor) {
    if (tensor == NULL) {
        return NULL;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return NULL;
    }

    cgrad_graph_node* node;
    int ret = cgrad_compute_graph_get_node(graph, tensor->node_id, &node);
    if (ret != CGRAD_SUCCESS) {
        return NULL;
    }

    return node->storage;
}

void cgrad_tensor_print(const cgrad_tensor* tensor) {
    if (tensor == NULL) {
        printf("Tensor: NULL\n");
        return;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        printf("Tensor: global graph not initialized\n");
        return;
    }

    cgrad_graph_node* node;
    int ret = cgrad_compute_graph_get_node(graph, tensor->node_id, &node);
    if (ret == CGRAD_SUCCESS) {
        printf("Op: %s\n", cgrad_op_type_to_string(node->op_info.type));
        printf("Storage: %s\n", node->storage ? "materialized" : "lazy");
        
        if (node->storage) {
            cgrad_storage_print(node->storage);
        }
    }
}

// ============================================================================
// Gradient Functions
// ============================================================================

int cgrad_tensor_set_requires_grad(cgrad_tensor* tensor, int requires_grad) {
    if (tensor == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    cgrad_graph_node* node;
    int ret = cgrad_compute_graph_get_node(graph, tensor->node_id, &node);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    node->requires_grad = requires_grad ? 1 : 0;
    return CGRAD_SUCCESS;
}

int cgrad_tensor_get_requires_grad(const cgrad_tensor* tensor, int* out_requires_grad) {
    if (tensor == NULL || out_requires_grad == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    cgrad_graph_node* node;
    int ret = cgrad_compute_graph_get_node(graph, tensor->node_id, &node);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    *out_requires_grad = node->requires_grad;
    return CGRAD_SUCCESS;
}

cgrad_storage* cgrad_tensor_get_grad(const cgrad_tensor* tensor) {
    if (tensor == NULL) {
        return NULL;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return NULL;
    }

    cgrad_graph_node* node;
    int ret = cgrad_compute_graph_get_node(graph, tensor->node_id, &node);
    if (ret != CGRAD_SUCCESS) {
        return NULL;
    }

    return node->grad_storage;
}

int cgrad_tensor_zero_grad(void) {
    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Delegate to compute graph
    return cgrad_compute_graph_zero_grad(graph);
}

int cgrad_tensor_backward(cgrad_tensor* tensor) {
    if (tensor == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_compute_graph* graph = get_global_graph();
    if (graph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Delegate to compute graph
    return cgrad_compute_graph_backward(graph, tensor->node_id);
}

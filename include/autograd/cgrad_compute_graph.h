#ifndef CGRAD_COMPUTE_GRAPH_H
#define CGRAD_COMPUTE_GRAPH_H

#include <uuid/uuid.h>
#include <graphviz/cgraph.h>
#include "storage/cgrad_storage_layout.h"
#include "storage/cgrad_storage.h"
#include "storage/cgrad_storage_backend.h"
#include "uthash.h"

#define MAX_NODE_INPUTS 16
#define MAX_GRAPH_NODES 1024

/**
 * @file cgrad_compute_graph.h
 * @brief Lazy computation graph for cgrad tensor library.
 * 
 * This module implements a lazy evaluation framework where tensors are nodes
 * in a computation graph. Operations build the graph without executing,
 * and execution happens on-demand with result caching.
 */

/**
 * @brief Enumeration of all supported operations in the compute graph.
 * 
 * Only non-leaf operations are included. Leaf nodes (inputs) are created
 * with materialized eager storage and are not represented by an operation type.
 */
typedef enum {
    CGRAD_OP_NONE = 0,            /**< No operation (used for leaf nodes) */
    
    // Element-wise binary operations (2 tensor inputs)
    CGRAD_OP_AXPY = 1,            /**< AXPY operation: c = alpha * a + b */
    
    // Linear algebra operations (2 tensor inputs)
    CGRAD_OP_GEMM = 3,            /**< Batched matrix multiplication (GEMM) */
    
    // Unary operations (1 tensor input)
    CGRAD_OP_TRANSPOSE = 4,       /**< Transpose along specified axes */
    CGRAD_OP_RESHAPE = 5,         /**< Reshape to new dimensions */
    CGRAD_OP_REDUCE_SUM = 6,      /**< Sum reduction along axes */
} cgrad_op_type;

/**
 * @brief Union containing operation-specific metadata.
 * The relevant field depends on the operation type.
 */
typedef union {
    struct {
        uint32_t perm[TENSOR_DIM];  /**< Permutation for transpose */
        int ndim;                   /**< Number of dimensions to permute */
    } transpose;                    /**< Metadata for CGRAD_OP_TRANSPOSE */
    
    struct {
        int32_t new_shape[TENSOR_DIM];  /**< Target shape for reshape */
        int ndim;                        /**< Number of dimensions */
    } reshape;                      /**< Metadata for CGRAD_OP_RESHAPE */
    
    struct {
        uint8_t mask[TENSOR_DIM];   /**< Reduction mask (1=reduce, 0=keep) */
        int ndim;                   /**< Number of dimensions */
    } reduce_sum;                   /**< Metadata for CGRAD_OP_REDUCE_SUM */
    
    struct {
        float alpha;                /**< Scalar multiplier for A*B */
        float beta;                 /**< Scalar multiplier for C */
    } gemm;                         /**< Metadata for CGRAD_OP_GEMM */
    
    struct {
        float alpha;                /**< Scalar multiplier for x in y = alpha*x + y */
    } axpy;                          /**< Metadata for CGRAD_OP_AXPY */
    
    float scalar;                   /**< For scalar operations */
} cgrad_op_metadata;

/**
 * @brief Information about an operation and its metadata.
 */
typedef struct {
    cgrad_op_type type;             /**< Type of operation */
    cgrad_op_metadata metadata;     /**< Operation-specific metadata */
} cgrad_op_info;

/**
 * @brief A node in the computation graph.
 * Represents an operation and its output tensor.
 * 
 * For leaf nodes (inputs with no incoming edges), the materialized eager storage
 * is cached here. For operation nodes, storage is NULL and is computed during execution.
 * 
 * Graph connectivity is determined via libcgraph edges rather than being cached here.
 * To find input nodes, query incoming edges using libcgraph API.
 */
typedef struct cgrad_graph_node {
    uuid_t node_id;                    /**< Unique identifier for this node */
    cgrad_op_info op_info;             /**< Operation type and metadata (unused for leaves) */
    cgrad_storage_layout layout;       /**< Shape of the output tensor */
    cgrad_storage* storage;            /**< For leaf: materialized eager storage; for ops: NULL or cached */
    cgrad_storage* grad_storage;       /**< Cached gradient (NULL if not computed) */
    void* ctx;                         /**< Operation context for caching intermediate results (NULL by default) */
    cgrad_storage_backend_type backend_type;  /**< Backend type of the node */
    int ref_count;                     /**< Reference count for memory management */
    int requires_grad;                 /**< 1 if gradients should be computed, 0 otherwise */
    UT_hash_handle hh;                 /**< Hash handle for uthash */
} cgrad_graph_node;

/**
 * @brief A directed acyclic graph (DAG) of computations.
 * Uses libcgraph as the backend graph implementation.
 * 
 * The graph itself is graph-centric and doesn't track specific output nodes.
 * Instead, any node can be materialized on demand, with results cached during execution
 * to enable efficient reuse of intermediate computations.
 */
typedef struct cgrad_compute_graph {
    uuid_t graph_id;                 /**< Unique identifier for this graph */
    
    // libcgraph backend
    Agraph_t* agraph;                /**< libcgraph graph object */
    
    // Metadata storage
    // Maps node_id (uuid) -> cgrad_graph_node via hash table
    cgrad_graph_node* node_metadata_table;  /**< Hash table: uuid_t -> cgrad_graph_node */
} cgrad_compute_graph;

// ============================================================================
// Graph Management Functions
// ============================================================================

/**
 * @brief Create a new empty computation graph.
 * @param graph Pointer to graph to initialize.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_create(cgrad_compute_graph* graph);

/**
 * @brief Get the node information for a graph node.
 * @param graph Compute graph.
 * @param node_id Node identifier.
 * @param out_node_info Pointer to output node info.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_get_node(
    const cgrad_compute_graph* graph,
    const uuid_t node_id,
    cgrad_graph_node** out_node_info
);

/**
 * @brief Get all input nodes (source nodes) for a given node.
 * @param graph Compute graph.
 * @param node_id Target node identifier.
 * @param out_input_node_ids Array to store input node IDs.
 * @param max_inputs Maximum number of inputs to return.
 * @param out_num_inputs Pointer to actual number of inputs found.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_get_inputs(
    const cgrad_compute_graph* graph,
    const uuid_t node_id,
    uuid_t* out_input_node_ids,
    int max_inputs,
    int* out_num_inputs
);

/**
 * @brief Free all resources associated with a computation graph.
 * @param graph Compute graph to free.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_free(cgrad_compute_graph* graph);

/**
 * @brief Get the number of nodes currently in the graph.
 * @param graph Compute graph.
 * @return Number of nodes in the graph.
 */
int cgrad_compute_graph_get_node_count(const cgrad_compute_graph* graph);

/**
 * @brief Free a single node and recursively free its inputs if their ref_count reaches zero.
 * 
 * This function is used internally by the reference counting system.
 * 
 * @param graph Compute graph.
 * @param node Node to free.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_free_node(cgrad_compute_graph* graph, cgrad_graph_node* node);

/**
 * @brief Decrement the reference count of a node and free it if count reaches zero.
 * 
 * This is the main entry point for reference counting. When a tensor is freed,
 * it should call this function to decrement the node's reference count.
 * If the count reaches zero, the node and its inputs are freed recursively.
 * 
 * @param graph Compute graph.
 * @param node_id ID of the node to decrement.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_decrement_ref(cgrad_compute_graph* graph, const uuid_t node_id);

/**
 * @brief Increment the reference count of a node.
 * 
 * This should be called when a new tensor reference to a node is created
 * (e.g., when copying a tensor).
 * 
 * @param graph Compute graph.
 * @param node_id ID of the node to increment.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_increment_ref(cgrad_compute_graph* graph, const uuid_t node_id);

// ============================================================================
// Node Management Functions
// ============================================================================

/**
 * @brief Add a leaf node (input) to the graph.
 * @param graph Compute graph.
 * @param layout Shape of the tensor.
 * @param storage Materialized eager storage.
 * @param out_node_id Output node ID.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_add_leaf(
    cgrad_compute_graph* graph,
    const cgrad_storage_layout* layout,
    cgrad_storage* storage,
    uuid_t out_node_id
);

/**
 * @brief Add an operation node to the graph.
 * @param graph Compute graph.
 * @param op_info Operation information.
 * @param layout Output shape of the operation.
 * @param input_node_ids Array of input node IDs.
 * @param num_inputs Number of inputs.
 * @param out_node_id Output node ID.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_add_op(
    cgrad_compute_graph* graph,
    const cgrad_op_info* op_info,
    const cgrad_storage_layout* layout,
    const uuid_t* input_node_ids,
    int num_inputs,
    uuid_t out_node_id
);

// ============================================================================
// Graph Visualization and Debugging
// ============================================================================

/**
 * @brief Export the computation graph to Graphviz DOT format.
 * 
 * Useful for visualization and debugging of graph structure.
 * Generated DOT can be visualized with: dot -Tpng graph.dot -o graph.png
 * 
 * @param graph Compute graph.
 * @param filename Path to output DOT file.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_to_dot(
    const cgrad_compute_graph* graph,
    const char* filename
);

/**
 * @brief Print human-readable summary of the computation graph.
 * @param graph Compute graph.
 */
void cgrad_compute_graph_print(const cgrad_compute_graph* graph);

/**
 * @brief Print information about a single node.
 * @param node Node to print.
 */
void cgrad_graph_node_print(const cgrad_graph_node* node);

/**
 * @brief Get string name of operation type.
 * @param op_type Operation type.
 * @return String name of operation.
 */
const char* cgrad_op_type_to_string(cgrad_op_type op_type);

// ============================================================================
// Graph Execution Functions
// ============================================================================

/**
 * @brief Execute forward pass for a computation subgraph.
 * 
 * This function:
 * 1. Performs topological sort to collect the dependency subgraph
 * 2. Executes operations in topological order, materializing and caching results
 * 
 * Leaf nodes (already materialized) are skipped. Operation nodes are executed
 * by fetching their input storages, applying the operation, and caching the
 * result storage in the node.
 * 
 * @param graph Compute graph containing the nodes.
 * @param target_node_id The node to execute (endpoint of the subgraph).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_forward(
    cgrad_compute_graph* graph,
    const uuid_t target_node_id
);

// ============================================================================
// Backward Pass Functions
// ============================================================================

/**
 * @brief Compute gradients by backpropagation through the computation graph.
 * 
 * This function:
 * 1. Initializes the gradient of the target tensor to 1.0
 * 2. Traverses the graph in reverse topological order
 * 3. For each node with requires_grad=1, computes gradients of inputs using the backward function
 * 4. Accumulates gradients when a node is used multiple times
 * 5. Stores gradients in each node's grad_storage field
 * 
 * The target tensor must have been executed (forward pass) before calling backward.
 * 
 * @param graph Compute graph containing the nodes.
 * @param target_node_id Target node (typically a scalar loss).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_backward(
    cgrad_compute_graph* graph,
    const uuid_t target_node_id
);

/**
 * @brief Zero out all gradients in the computation graph.
 * 
 * This should be called before each backward pass in training loops.
 * 
 * @param graph Compute graph.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_zero_grad(cgrad_compute_graph* graph);

/**
 * @brief Zero out the gradient of a specific node.
 * 
 * This function sets the gradient storage of the given node to zero.
 * If the gradient doesn't exist yet, this function does nothing and returns success.
 * 
 * @param graph Compute graph.
 * @param node_id Node identifier.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_zero_grad_node(cgrad_compute_graph* graph, const uuid_t node_id);

/**
 * @brief Set the requires_grad flag for a node.
 * 
 * @param graph Compute graph.
 * @param node_id Node identifier.
 * @param requires_grad 1 to enable gradient computation, 0 to disable.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_set_requires_grad(
    cgrad_compute_graph* graph,
    const uuid_t node_id,
    int requires_grad
);

/**
 * @brief Get the storage of a node.
 * 
 * Returns the storage if the node has been computed, NULL otherwise.
 * 
 * @param graph Compute graph.
 * @param node_id Node identifier.
 * @return Const pointer to storage, or NULL if not computed.
 */
const cgrad_storage* cgrad_compute_graph_get_storage(
    const cgrad_compute_graph* graph,
    const uuid_t node_id
);

/**
 * @brief Get the gradient storage of a node.
 * 
 * Returns the gradient storage if it exists, NULL otherwise.
 * 
 * @param graph Compute graph.
 * @param node_id Node identifier.
 * @return Const pointer to gradient storage, or NULL if not available.
 */
const cgrad_storage* cgrad_compute_graph_get_grad_storage(
    const cgrad_compute_graph* graph,
    const uuid_t node_id
);

#endif // CGRAD_COMPUTE_GRAPH_H

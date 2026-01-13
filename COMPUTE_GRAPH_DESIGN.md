# Lazy Compute Graph Design for cgrad

## Overview

This document describes the design of a lazy compute graph framework for the cgrad tensor library. The framework builds on top of the existing eager storage implementation, enabling deferred computation and optimization opportunities through graph analysis before execution.

### Key Concepts

- **Lazy Evaluation**: Tensors are nodes in a computation graph rather than eager values
- **Compute Graph**: Directed acyclic graph (DAG) representing the computation pipeline
- **Node**: Represents a tensor and the operation that produced it
- **Edge**: Represents data flow with argument index information
- **Execution**: Convert the lazy graph to eager execution using the storage backend

## Architecture Decision: Per-Tensor Graph Tracking with Merging

We adopt a **per-tensor graph tracking approach** where:

1. Each lazy tensor maintains a reference to its computation graph
2. When combining tensors from different graphs, graphs are merged
3. This allows lazy tensors to be composed and passed around independently
4. Leaf tensors (inputs) reference graphs without dependencies
5. During execution, all connected graphs are materialized using the eager storage backend

**Advantages**:
- Modular composition of tensor operations
- Natural representation of partial computations
- Cleaner API without requiring global state
- Better support for multiple independent computations

## Core Data Structures

### 1. Operation Enum

All supported operations are defined as an enum. Each operation specifies its arity (number of tensor inputs) and any additional metadata requirements.

Only operations are defined here; leaf nodes (inputs) are handled separately with materialized storage.

```c
/**
 * @brief Enumeration of all supported operations in the compute graph.
 * 
 * Only non-leaf operations are included. Leaf nodes (inputs) are created
 * with materialized eager storage and are not represented by an operation type.
 */
typedef enum {
    // Element-wise binary operations (2 tensor inputs)
    CGRAD_OP_ADD = 1,             /**< Element-wise addition: c = a + b */
    CGRAD_OP_SUB = 2,             /**< Element-wise subtraction: c = a - b */
    
    // Linear algebra operations (2 tensor inputs)
    CGRAD_OP_GEMM = 3,            /**< Batched matrix multiplication (GEMM) */
    
    // Unary operations (1 tensor input)
    CGRAD_OP_TRANSPOSE = 4,       /**< Transpose along specified axes */
    CGRAD_OP_RESHAPE = 5,         /**< Reshape to new dimensions */
    CGRAD_OP_REDUCE_SUM = 6,      /**< Sum reduction along axes */

} cgrad_op_type;
```

### 2. Operation Metadata

Each operation may require additional metadata beyond input tensors. This is stored as a union to minimize memory overhead.

```c
/**
 * @brief Union containing operation-specific metadata.
 * The relevant field depends on the operation type.
 */
typedef union {
    struct {
        uint32_t perm[TENSOR_DIM];  /**< Permutation for transpose */
        int ndim;                   /**< Number of dimensions to permute */
    } transpose;
    
    struct {
        int32_t new_shape[TENSOR_DIM];  /**< Target shape for reshape */
        int ndim;                        /**< Number of dimensions */
    } reshape;
    
    struct {
        uint8_t mask[TENSOR_DIM];   /**< Reduction mask (1=reduce, 0=keep) */
        int ndim;                   /**< Number of dimensions */
    } reduce_sum;
    
    struct {
        int axis;                   /**< Axis for softmax */
    } softmax;
    
    struct {
        uint8_t mask[TENSOR_DIM];   /**< Reduction mask */
        int ndim;                   /**< Number of dimensions */
    } mean;
    
    float scalar;                   /**< For scalar operations */
    
} cgrad_op_metadata;

/**
 * @brief Information about an operation and its metadata.
 */
typedef struct {
    cgrad_op_type type;             /**< Type of operation */
    cgrad_op_metadata metadata;     /**< Operation-specific metadata */
} cgrad_op_info;
```

### 3. Compute Graph Node

Each node in the graph represents a single operation and its output tensor. For leaf nodes (inputs), 
the materialized storage is cached in the node.

```c
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
    cgrad_storage* storage;            /**< For leaf: materialized eager storage; for ops: NULL (internal only) */
} cgrad_graph_node;
```

### 4. Compute Graph Edges

Edges represent data dependencies and are managed by libcgraph. Each edge has a "slot" attribute 
that specifies which argument position the source tensor fills.

For example, in a subtraction operation c = a - b:
- Edge from a to subtract node has slot = "0" (first input)
- Edge from b to subtract node has slot = "1" (second input)

The slot value is set using libcgraph's `agsafeset()` function:

```c
// Create edge from source node to target node
Agedge_t* edge = agedge(agraph, source_node, target_node, NULL, 1);

// Set the slot attribute (which input argument this is)
agsafeset(edge, "slot", "0", -1);  // This is the first input
```

To retrieve the slot attribute:

```c
char* slot_str = agget(edge, "slot");
int slot = atoi(slot_str);  // Convert "0" or "1" to integer
```

The string-to-int conversion is efficient for single-digit values and keeps the design simple
by leveraging libcgraph's built-in attribute system without additional hash tables.

### 5. Computation Graph

The graph container manages nodes, edges, and provides query operations. It uses `libcgraph` as the underlying implementation.

```c
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
    void* node_metadata_table;       /**< Hash table: uuid_t -> cgrad_graph_node */
    
} cgrad_compute_graph;
```

### 6. Tensor

A tensor is a reference to a computation graph node. For leaf nodes, it wraps materialized storage internally.
For operation nodes, it represents a deferred value.

```c
/**
 * @brief A tensor in the computation graph.
 * 
 * A tensor represents a value in the computation graph. For leaf nodes (inputs),
 * materialized eager storage is managed internally via the graph node's cached_storage.
 * For operation nodes, the tensor represents a deferred computation.
 * 
 * Users do not interact with cgrad_storage directly; all operations go through
 * the tensor interface which manages storage internally.
 */
typedef struct {
    uuid_t node_id;                  /**< ID of the node producing this tensor */
    cgrad_compute_graph* graph;      /**< Pointer to the compute graph */
    cgrad_storage_layout layout;     /**< Shape and layout of the tensor */
} cgrad_tensor;
```

## Operation Graph Construction

### Creating Input Tensors

Input tensors are leaf nodes with materialized data. The user interacts only with the tensor interface:

```c
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
int cgrad_tensor_init(cgrad_tensor* tensor,
                      const uint32_t* shape, int ndim,
                      cgrad_storage_backend_type backend_type);

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
```

**Usage Pattern:**
```c
// Create input tensors (materialized storage is internal)
uint32_t shape_a[] = {3, 4};
uint32_t shape_b[] = {3, 4};

cgrad_tensor a, b;
cgrad_tensor_init(&a, shape_a, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
cgrad_tensor_init(&b, shape_b, 2, CGRAD_STORAGE_BACKEND_F32_CPU);

// Fill with data (storage hidden from user)
cgrad_tensor_fill_rand(&a);
cgrad_tensor_fill_rand(&b);

// Now a and b can be used in operations
cgrad_tensor c;
cgrad_tensor_add(&a, &b, &c);
```

### Binary Operations

```c
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
int cgrad_tensor_add(const cgrad_tensor* a,
                     const cgrad_tensor* b,
                     cgrad_tensor* out_tensor);

/**
 * @brief Element-wise subtraction: out = a - b
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param out_tensor Pointer to output tensor.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_sub(const cgrad_tensor* a,
                     const cgrad_tensor* b,
                     cgrad_tensor* out_tensor);

/**
 * @brief Batched matrix multiplication: out = a @ b
 * @param a First input tensor (shape: ..., m, k).
 * @param b Second input tensor (shape: ..., k, n).
 * @param out_tensor Pointer to output tensor (shape: ..., m, n).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_gemm(const cgrad_tensor* a,
                      const cgrad_tensor* b,
                      cgrad_tensor* out_tensor);
```

### Unary Operations

```c
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
int cgrad_tensor_transpose(const cgrad_tensor* tensor,
                           const uint32_t* perm, int ndim,
                           cgrad_tensor* out_tensor);

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
int cgrad_tensor_reshape(const cgrad_tensor* tensor,
                         const int32_t* new_shape, int ndim,
                         cgrad_tensor* out_tensor);

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
int cgrad_tensor_reduce_sum(const cgrad_tensor* tensor,
                            const uint8_t* mask, int ndim,
                            cgrad_tensor* out_tensor);
```

## Graph Execution

### Materialization

Once a computation graph is defined, tensors can be materialized on-demand by executing the graph.
Results are cached during execution to enable efficient reuse of intermediate computations.
All storage is managed internally.

```c
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
```

**Internal Storage Management:**

After execution, the tensor's underlying storage is cached in the computation graph node. Future calls to `cgrad_tensor_execute()` will detect cached results and skip recomputation from those cached nodes, significantly improving performance when:
- The same tensor is executed multiple times
- Multiple tensors share common subgraphs
- Part of a graph has already been computed

**Usage Pattern:**
```c
// Create and initialize inputs
cgrad_tensor a, b;
cgrad_tensor_init(&a, shape_a, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
cgrad_tensor_init(&b, shape_b, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
cgrad_tensor_fill_rand(&a);
cgrad_tensor_fill_rand(&b);

// Build computation graph
cgrad_tensor c;
cgrad_tensor_add(&a, &b, &c);

// Execute to materialize results
cgrad_tensor_execute(&c);

// c now has materialized storage in the graph
// Can use c in further operations without re-executing
```

## Graph Management and Introspection

```c
/**
 * @brief Merge two computation graphs.
 * 
 * When operations combine tensors from different graphs,
 * this function merges them into a single graph.
 * 
 * @param graph1 First graph (will receive merged nodes).
 * @param graph2 Second graph (nodes copied into graph1).
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_merge(cgrad_compute_graph* graph1,
                              cgrad_compute_graph* graph2);

/**
 * @brief Get the node information for a graph node.
 * @param graph Compute graph.
 * @param node_id Node identifier.
 * @param out_node_info Pointer to output node info.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_get_node(const cgrad_compute_graph* graph,
                                 uuid_t node_id,
                                 cgrad_graph_node* out_node_info);

/**
 * @brief Get all input nodes (source nodes) for a given node.
 * @param graph Compute graph.
 * @param node_id Target node identifier.
 * @param out_input_node_ids Array to store input node IDs.
 * @param max_inputs Maximum number of inputs to return.
 * @param out_num_inputs Pointer to actual number of inputs found.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_get_inputs(const cgrad_compute_graph* graph,
                                   uuid_t node_id,
                                   uuid_t* out_input_node_ids,
                                   int max_inputs,
                                   int* out_num_inputs);

/**
 * @brief Perform topological sort of the computation graph.
 * 
 * Returns node IDs in topological order suitable for execution.
 * 
 * @param graph Compute graph.
 * @param out_sorted_node_ids Array to store sorted node IDs.
 * @param max_nodes Maximum number of nodes.
 * @param out_num_nodes Pointer to actual number of nodes.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_topological_sort(const cgrad_compute_graph* graph,
                                         uuid_t* out_sorted_node_ids,
                                         int max_nodes,
                                         int* out_num_nodes);

/**
 * @brief Free all resources associated with a computation graph.
 * @param graph Compute graph to free.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_compute_graph_free(cgrad_compute_graph* graph);

/**
 * @brief Free a tensor (does not free the underlying graph).
 * @param tensor Tensor to free.
 * @return CGRAD_SUCCESS on success, error code otherwise.
 */
int cgrad_tensor_free(cgrad_tensor* tensor);
```

## Graph Visualization and Debugging

```c
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
int cgrad_compute_graph_to_dot(const cgrad_compute_graph* graph,
                               const char* filename);

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
```

## Usage Example

```c
// Create input tensors
uint32_t shape_a[] = {3, 4};
uint32_t shape_b[] = {3, 4};

cgrad_tensor a, b;
cgrad_tensor_init(&a, shape_a, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
cgrad_tensor_init(&b, shape_b, 2, CGRAD_STORAGE_BACKEND_F32_CPU);

// Fill with data
cgrad_tensor_fill_rand(&a);
cgrad_tensor_fill_rand(&b);

// Build computation graph lazily
cgrad_tensor c;
cgrad_tensor_add(&a, &b, &c);  // c = a + b

// More operations
cgrad_tensor d;
cgrad_tensor_transpose(&c, (uint32_t[]){1, 0}, 2, &d);  // d = c^T

// Visualize the graph (optional)
cgrad_compute_graph_to_dot(d.graph, "computation.dot");

// Execute the graph - storage is managed internally
cgrad_tensor_execute(&d);

// d now contains materialized results
// All storage is hidden from the user

// Cleanup
cgrad_tensor_free(&a);
cgrad_tensor_free(&b);
cgrad_tensor_free(&c);
cgrad_tensor_free(&d);
```

## Implementation Considerations

### Graph Backend (libcgraph)

The framework uses `libcgraph` (from Graphviz) as the underlying graph implementation:

```c
#include <graphviz/cgraph.h>

// libcgraph provides:
// - Agraph_t: Graph container
// - Agnode_t: Nodes
// - Agedge_t: Edges
// - agopen, agclose: Create/destroy graphs
// - agnode, agedge: Create nodes and edges
// - agfindnode, agfindedge: Look up elements
// - aginit, agread, agwrite: I/O and parsing
```

### Memory Management

1. **Node Metadata**: Stored in a hash table (`uthash.h`) mapping `uuid_t` -> `cgrad_graph_node`
2. **Graph Merging**: Copy node metadata from source to target graph
3. **Execution Cache**: Temporary cache of materialized tensors during execution

### Shape Inference

- Input shapes must be explicitly provided
- Output shapes are inferred from input shapes and operation semantics
- Transpose, reshape, and reduction operations update shape accordingly
- Broadcasting is applied for binary operations

### Graph Merging Strategy

When combining tensors from different graphs:

```
Graph 1: Input_A -> Transpose -> [Node_1]
Graph 2: Input_B -> ReLU -> [Node_2]

Operation: Add(Node_1, Node_2)
Result: Merge Graph 2 into Graph 1
         Add node connects [Node_1] and [Node_2]
         New graph has both Input_A and Input_B as roots
```

## Future Extensions

1. **Gradient Computation**: Automatic differentiation through the graph
2. **Graph Optimization**: Constant folding, operation fusion, dead code elimination
3. **Kernel Fusion**: Combine multiple operations into single kernels
4. **Memory Optimization**: Compute optimal memory layout and reuse strategies
5. **Device Placement**: Distribute computation across multiple devices
6. **Parallel Execution**: Execute independent subgraphs concurrently
7. **Dynamic Shapes**: Support shape computation at runtime

## Error Handling

All functions return error codes defined in `cgrad_errors.h`:

```c
#define CGRAD_SUCCESS 0
#define CGRAD_GRAPH_ERR_SHAPE_MISMATCH -1400
#define CGRAD_GRAPH_ERR_INVALID_OPERATION -1401
#define CGRAD_GRAPH_ERR_GRAPH_MERGE_FAILED -1402
#define CGRAD_GRAPH_ERR_TOPOLOGICAL_SORT_FAILED -1403
#define CGRAD_GRAPH_ERR_EXECUTION_FAILED -1404
```

Callers should always check return codes before proceeding.

## Thread Safety

The current design is **not thread-safe**. For concurrent access:
- Use separate graphs and tensors per thread
- Or implement mutex-protected access to shared graphs
- Consider thread-local storage for execution caches

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Create input tensor | O(1) | Single node in new graph |
| Binary operation | O(1) | Add one node and edges |
| Unary operation | O(1) | Add one node and edge |
| Graph merge | O(N+E) | Copy all nodes and edges |
| Topological sort | O(N+E) | Using Kahn's algorithm |
| Graph execution | O(N+E) + data transfer | Traversal + eager backend ops |

Where N = number of nodes, E = number of edges

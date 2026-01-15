#include "autograd/cgrad_compute_graph.h"
#include "autograd/cgrad_ops.h"
#include "cgrad_errors.h"
#include "uthash.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Convert UUID to string for libcgraph node names.
 */
static void uuid_to_string(const uuid_t uuid, char* out_str) {
    uuid_unparse(uuid, out_str);
}

/**
 * @brief Find node in metadata table by UUID.
 */
static cgrad_graph_node* find_node_metadata(cgrad_compute_graph* graph, uuid_t node_id) {
    cgrad_graph_node* node = NULL;
    HASH_FIND(hh, graph->node_metadata_table, node_id, sizeof(uuid_t), node);
    return node;
}

/**
 * @brief Add node to metadata table.
 */
static int add_node_metadata(cgrad_compute_graph* graph, cgrad_graph_node* node) {
    cgrad_graph_node* existing = find_node_metadata(graph, node->node_id);
    if (existing != NULL) {
        return CGRAD_GRAPH_ERR_INVALID_NODE;  // Node already exists
    }
    HASH_ADD(hh, graph->node_metadata_table, node_id, sizeof(uuid_t), node);
    return CGRAD_SUCCESS;
}

// ============================================================================
// Backward Pass
// ============================================================================

int cgrad_compute_graph_backward(
    cgrad_compute_graph* graph,
    const uuid_t target_node_id
) {
    if (graph == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    // Get target node
    cgrad_graph_node* target_node;
    int ret = cgrad_compute_graph_get_node(graph, target_node_id, &target_node);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    // Check that forward pass has been executed
    if (target_node->storage == NULL) {
        return CGRAD_GRAPH_ERR_FORWARD_NOT_EXECUTED;
    }

    // Topological sort
    uuid_t sorted_ids[MAX_GRAPH_NODES];
    int num_nodes;
    ret = cgrad_compute_graph_topological_sort(
        graph, target_node_id, sorted_ids, MAX_GRAPH_NODES, &num_nodes
    );
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    // Initialize target gradient to 1.0
    if (target_node->grad_storage == NULL) {
        target_node->grad_storage = (cgrad_storage*)calloc(1, sizeof(cgrad_storage));
        if (target_node->grad_storage == NULL) {
            return CGRAD_GRAPH_ERR_ALLOC_FAILED;
        }
        ret = cgrad_storage_init(
            target_node->grad_storage,
            target_node->layout.shape,
            TENSOR_DIM,
            target_node->backend_type
        );
        if (ret != CGRAD_SUCCESS) {
            free(target_node->grad_storage);
            target_node->grad_storage = NULL;
            return ret;
        }
    }
    ret = cgrad_storage_fill(target_node->grad_storage, 1.0f);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    // Backward pass in reverse topological order
    for (int i = num_nodes - 1; i >= 0; i--) {
        cgrad_graph_node* node;
        ret = cgrad_compute_graph_get_node(graph, sorted_ids[i], &node);
        if (ret != CGRAD_SUCCESS) {
            return ret;
        }

        // Skip if doesn't require gradients
        if (!node->requires_grad) continue;

        // Skip leaf nodes (no backward to compute)
        if (node->op_info.type == CGRAD_OP_NONE) continue;

        // Get incoming gradient
        if (node->grad_storage == NULL) {
            continue;
        }

        // Get operation descriptor
        const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(node->op_info.type);
        if (op_desc == NULL || op_desc->backward == NULL) {
            return CGRAD_GRAPH_ERR_BACKWARD_NOT_IMPLEMENTED;
        }

        // Get input nodes and storages
        uuid_t input_ids[MAX_NODE_INPUTS];
        int num_inputs;
        ret = cgrad_compute_graph_get_inputs(graph, node->node_id, input_ids, MAX_NODE_INPUTS, &num_inputs);
        if (ret != CGRAD_SUCCESS) {
            return ret;
        }

        cgrad_storage* input_storages[MAX_NODE_INPUTS];
        cgrad_graph_node* input_nodes[MAX_NODE_INPUTS];
        cgrad_storage* grad_inputs[MAX_NODE_INPUTS];
        int input_requires_grad[MAX_NODE_INPUTS];

        for (int j = 0; j < num_inputs; j++) {
            ret = cgrad_compute_graph_get_node(graph, input_ids[j], &input_nodes[j]);
            if (ret != CGRAD_SUCCESS) {
                return ret;
            }
            input_storages[j] = input_nodes[j]->storage;
            input_requires_grad[j] = input_nodes[j]->requires_grad;

            // Allocate gradient storage for inputs if needed
            if (input_nodes[j]->requires_grad && input_nodes[j]->grad_storage == NULL) {
                input_nodes[j]->grad_storage = (cgrad_storage*)calloc(1, sizeof(cgrad_storage));
                if (input_nodes[j]->grad_storage == NULL) {
                    return CGRAD_GRAPH_ERR_ALLOC_FAILED;
                }
                ret = cgrad_storage_init(
                    input_nodes[j]->grad_storage,
                    input_nodes[j]->layout.shape,
                    TENSOR_DIM,
                    input_nodes[j]->backend_type
                );
                if (ret != CGRAD_SUCCESS) {
                    return ret;
                }
                ret = cgrad_storage_fill(input_nodes[j]->grad_storage, 0.0f);
                if (ret != CGRAD_SUCCESS) {
                    return ret;
                }
            }
            grad_inputs[j] = input_nodes[j]->grad_storage;
        }

        // Call backward function
        ret = op_desc->backward(
            input_storages,
            num_inputs,
            node->storage,
            node->grad_storage,
            &node->op_info.metadata,
            node->ctx,
            grad_inputs,
            input_requires_grad
        );
        if (ret != CGRAD_SUCCESS) {
            return ret;
        }

        // Free context if operation has a free_ctx function
        if (node->ctx != NULL && op_desc->free_ctx != NULL) {
            op_desc->free_ctx(node->ctx);
            node->ctx = NULL;
        }
    }

    return CGRAD_SUCCESS;
}

int cgrad_compute_graph_zero_grad(cgrad_compute_graph* graph) {
    if (graph == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    // Iterate through all nodes and zero their gradients
    cgrad_graph_node *node, *tmp;
    HASH_ITER(hh, graph->node_metadata_table, node, tmp) {
        if (node->grad_storage != NULL) {
            int ret = cgrad_storage_fill(node->grad_storage, 0.0f);
            if (ret != CGRAD_SUCCESS) {
                return ret;
            }
        }
    }

    return CGRAD_SUCCESS;
}

int cgrad_compute_graph_set_requires_grad(
    cgrad_compute_graph* graph,
    const uuid_t node_id,
    int requires_grad
) {
    if (graph == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_graph_node* node;
    int ret = cgrad_compute_graph_get_node(graph, node_id, &node);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    node->requires_grad = requires_grad;
    return CGRAD_SUCCESS;
}

// ============================================================================
// Graph Management Functions
// ============================================================================

int cgrad_compute_graph_create(cgrad_compute_graph* graph) {
    if (graph == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    // Generate graph ID
    uuid_generate(graph->graph_id);

    // Create libcgraph directed graph
    char graph_name[64];
    uuid_to_string(graph->graph_id, graph_name);
    graph->agraph = agopen(graph_name, Agdirected, NULL);
    if (graph->agraph == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Initialize edge attributes
    agattr(graph->agraph, AGEDGE, "slot", "0");
    agattr(graph->agraph, AGNODE, "type", "");
    agattr(graph->agraph, AGNODE, "op", "");

    // Initialize metadata table
    graph->node_metadata_table = NULL;

    return CGRAD_SUCCESS;
}

int cgrad_compute_graph_get_node(
    const cgrad_compute_graph* graph,
    const uuid_t node_id,
    cgrad_graph_node** out_node_info
) {
    if (graph == NULL || out_node_info == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_graph_node* node = NULL;
    HASH_FIND(hh, graph->node_metadata_table, node_id, sizeof(uuid_t), node);
    
    if (node == NULL) {
        return CGRAD_GRAPH_ERR_NODE_NOT_FOUND;
    }

    *out_node_info = node;
    return CGRAD_SUCCESS;
}

int cgrad_compute_graph_get_inputs(
    const cgrad_compute_graph* graph,
    const uuid_t node_id,
    uuid_t* out_input_node_ids,
    int max_inputs,
    int* out_num_inputs
) {
    if (graph == NULL || out_input_node_ids == NULL || out_num_inputs == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    // Find the node in libcgraph
    char node_name[64];
    uuid_to_string(node_id, node_name);
    Agnode_t* ag_node = agnode(graph->agraph, node_name, 0);
    if (ag_node == NULL) {
        return CGRAD_GRAPH_ERR_NODE_NOT_FOUND;
    }

    // Collect incoming edges with slot information
    typedef struct {
        uuid_t node_id;
        int slot;
    } input_info;
    
    input_info inputs[MAX_NODE_INPUTS];
    int count = 0;

    Agedge_t* e;
    for (e = agfstin(graph->agraph, ag_node); e; e = agnxtin(graph->agraph, e)) {
        if (count >= MAX_NODE_INPUTS) break;
        
        char* src_name = agnameof(agtail(e));
        char* slot_str = agget(e, "slot");
        int slot = slot_str ? atoi(slot_str) : count;

        uuid_parse(src_name, inputs[count].node_id);
        inputs[count].slot = slot;
        count++;
    }

    // Sort by slot to get correct order
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (inputs[j].slot < inputs[i].slot) {
                input_info tmp = inputs[i];
                inputs[i] = inputs[j];
                inputs[j] = tmp;
            }
        }
    }

    // Copy to output
    int num_to_copy = count < max_inputs ? count : max_inputs;
    for (int i = 0; i < num_to_copy; i++) {
        uuid_copy(out_input_node_ids[i], inputs[i].node_id);
    }

    *out_num_inputs = count;
    return CGRAD_SUCCESS;
}

int cgrad_compute_graph_topological_sort(
    const cgrad_compute_graph* graph,
    const uuid_t target_node_id,
    uuid_t* out_sorted_node_ids,
    int max_nodes,
    int* out_num_nodes
) {
    if (graph == NULL || out_sorted_node_ids == NULL || out_num_nodes == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    // Find target node
    char target_name[64];
    uuid_to_string(target_node_id, target_name);
    Agnode_t* target = agnode(graph->agraph, target_name, 0);
    if (target == NULL) {
        return CGRAD_GRAPH_ERR_NODE_NOT_FOUND;
    }

    // BFS/DFS to find all dependencies
    uuid_t visited[MAX_GRAPH_NODES];
    int visited_count = 0;
    uuid_t queue[MAX_GRAPH_NODES];
    int queue_start = 0, queue_end = 0;

    uuid_copy(queue[queue_end++], target_node_id);

    while (queue_start < queue_end) {
        uuid_t current_id;
        uuid_copy(current_id, queue[queue_start++]);

        // Check if already visited
        int already_visited = 0;
        for (int i = 0; i < visited_count; i++) {
            if (uuid_compare(visited[i], current_id) == 0) {
                already_visited = 1;
                break;
            }
        }
        if (already_visited) continue;

        uuid_copy(visited[visited_count++], current_id);

        // Get inputs
        uuid_t inputs[MAX_NODE_INPUTS];
        int num_inputs;
        int ret = cgrad_compute_graph_get_inputs(graph, current_id, inputs, MAX_NODE_INPUTS, &num_inputs);
        if (ret != CGRAD_SUCCESS && ret != CGRAD_GRAPH_ERR_NODE_NOT_FOUND) {
            return ret;
        }

        // Add inputs to queue
        for (int i = 0; i < num_inputs && i < MAX_NODE_INPUTS; i++) {
            if (queue_end < MAX_GRAPH_NODES) {
                uuid_copy(queue[queue_end++], inputs[i]);
            }
        }
    }

    // Now perform topological sort on visited nodes
    // Simple approach: repeatedly find nodes with no unprocessed dependencies
    int processed[MAX_GRAPH_NODES] = {0};
    int sort_count = 0;

    while (sort_count < visited_count && sort_count < max_nodes) {
        int found = 0;
        for (int i = 0; i < visited_count; i++) {
            if (processed[i]) continue;

            // Check if all dependencies are processed
            uuid_t inputs[MAX_NODE_INPUTS];
            int num_inputs;
            cgrad_compute_graph_get_inputs(graph, visited[i], inputs, MAX_NODE_INPUTS, &num_inputs);

            int all_deps_done = 1;
            for (int j = 0; j < num_inputs; j++) {
                int dep_processed = 0;
                for (int k = 0; k < visited_count; k++) {
                    if (uuid_compare(visited[k], inputs[j]) == 0) {
                        if (!processed[k]) {
                            all_deps_done = 0;
                            break;
                        }
                        dep_processed = 1;
                        break;
                    }
                }
                if (!all_deps_done) break;
            }

            if (all_deps_done) {
                uuid_copy(out_sorted_node_ids[sort_count++], visited[i]);
                processed[i] = 1;
                found = 1;
            }
        }

        if (!found && sort_count < visited_count) {
            return CGRAD_GRAPH_ERR_TOPOLOGICAL_SORT_FAILED;
        }
    }

    *out_num_nodes = sort_count;
    return CGRAD_SUCCESS;
}

int cgrad_compute_graph_free(cgrad_compute_graph* graph) {
    if (graph == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    // Free all node metadata and their storage
    cgrad_graph_node *node, *tmp;
    HASH_ITER(hh, graph->node_metadata_table, node, tmp) {
        HASH_DEL(graph->node_metadata_table, node);
        
        
        // Free storage if it exists
        if (node->storage != NULL) {
            cgrad_storage_free(node->storage);
            free(node->storage);
        }

        // Free storage if it exists
        if (node->grad_storage != NULL) {
            cgrad_storage_free(node->grad_storage);
            free(node->grad_storage);
        }

        free(node);
    }

    // Free libcgraph
    if (graph->agraph != NULL) {
        agclose(graph->agraph);
    }

    return CGRAD_SUCCESS;
}

int cgrad_compute_graph_get_node_count(const cgrad_compute_graph* graph) {
    if (graph == NULL) {
        return 0;
    }
    
    return HASH_COUNT(graph->node_metadata_table);
}

// ============================================================================
// Node Management Functions
// ============================================================================

int cgrad_compute_graph_add_leaf(
    cgrad_compute_graph* graph,
    const cgrad_storage_layout* layout,
    cgrad_storage* storage,
    uuid_t out_node_id
) {
    if (graph == NULL || layout == NULL || storage == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    // Create node metadata
    cgrad_graph_node* node = (cgrad_graph_node*)malloc(sizeof(cgrad_graph_node));
    if (node == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Allocate memory for storage and create a shallow copy
    // This ensures the graph owns the storage memory
    cgrad_storage* node_storage = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    if (node_storage == NULL) {
        free(node);
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    int ret = cgrad_storage_shallow_copy(storage, node_storage);
    if (ret != CGRAD_SUCCESS) {
        free(node_storage);
        free(node);
        return ret;
    }

    uuid_generate(node->node_id);
    node->op_info.type = CGRAD_OP_NONE;
    node->layout = *layout;
    node->storage = node_storage;
    node->grad_storage = NULL;  // Initialize gradient storage
    node->ctx = NULL;           // Initialize context
    
    // Track backend type from storage
    if (node_storage->backend == NULL) {
        free(node_storage);
        free(node);
        return CGRAD_ERR_NULL_POINTER;
    }
    node->backend_type = node_storage->backend->type;
    node->ref_count = 1;        // Initialize reference count
    node->requires_grad = 1;    // Default: leaf nodes require gradients

    // Add to metadata table
    ret = add_node_metadata(graph, node);
    if (ret != CGRAD_SUCCESS) {
        free(node_storage);
        free(node);
        return ret;
    }

    // Add to libcgraph
    char node_name[64];
    uuid_to_string(node->node_id, node_name);
    Agnode_t* ag_node = agnode(graph->agraph, node_name, 1);
    if (ag_node == NULL) {
        HASH_DEL(graph->node_metadata_table, node);
        free(node);
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Set node attributes
    agsafeset(ag_node, "type", "leaf", "");

    uuid_copy(out_node_id, node->node_id);
    return CGRAD_SUCCESS;
}

int cgrad_compute_graph_add_op(
    cgrad_compute_graph* graph,
    const cgrad_op_info* op_info,
    const cgrad_storage_layout* layout,
    const uuid_t* input_node_ids,
    int num_inputs,
    uuid_t out_node_id
) {
    if (graph == NULL || op_info == NULL || layout == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    // Validate number of inputs
    if (num_inputs > MAX_NODE_INPUTS) {
        return CGRAD_GRAPH_ERR_TOO_MANY_INPUTS;
    }

    // Validate backend consistency across inputs
    cgrad_storage_backend_type backend_type;
    int backend_initialized = 0;
    
    for (int i = 0; i < num_inputs; i++) {
        cgrad_graph_node* input_node;
        int ret = cgrad_compute_graph_get_node(graph, input_node_ids[i], &input_node);
        if (ret != CGRAD_SUCCESS) {
            return ret;
        }
        
        if (!backend_initialized) {
            backend_type = input_node->backend_type;
            backend_initialized = 1;
        } else {
            // Check if this input has a different backend
            if (input_node->backend_type != backend_type) {
                return CGRAD_GRAPH_ERR_BACKEND_MISMATCH;
            }
        }
    }

    // Create node metadata
    cgrad_graph_node* node = (cgrad_graph_node*)malloc(sizeof(cgrad_graph_node));
    if (node == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    uuid_generate(node->node_id);
    node->op_info = *op_info;
    node->layout = *layout;
    node->storage = NULL;       // Not computed yet
    node->grad_storage = NULL;  // Initialize gradient storage
    node->ctx = NULL;           // Initialize context
    node->backend_type = backend_type;  // Inherit backend from inputs
    node->ref_count = 1;        // Initialize reference count
    
    // Inherit requires_grad from inputs: if ANY input requires grad, so does this node
    node->requires_grad = 0;
    for (int i = 0; i < num_inputs; i++) {
        cgrad_graph_node* input_node;
        int ret = cgrad_compute_graph_get_node(graph, input_node_ids[i], &input_node);
        if (ret == CGRAD_SUCCESS && input_node->requires_grad) {
            node->requires_grad = 1;
            break;
        }
    }

    // Add to metadata table
    int ret = add_node_metadata(graph, node);
    if (ret != CGRAD_SUCCESS) {
        free(node);
        return ret;
    }

    // Add to libcgraph
    char node_name[64];
    uuid_to_string(node->node_id, node_name);
    Agnode_t* ag_node = agnode(graph->agraph, node_name, 1);
    if (ag_node == NULL) {
        HASH_DEL(graph->node_metadata_table, node);
        free(node);
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Set node attributes
    agsafeset(ag_node, "type", "op", "");
    agsafeset(ag_node, "op", cgrad_op_type_to_string(op_info->type), "");

    // Add edges from inputs and increment their reference counts
    for (int i = 0; i < num_inputs; i++) {
        char input_name[64];
        uuid_to_string(input_node_ids[i], input_name);
        Agnode_t* input_node = agnode(graph->agraph, input_name, 0);
        if (input_node == NULL) {
            return CGRAD_GRAPH_ERR_NODE_NOT_FOUND;
        }

        Agedge_t* edge = agedge(graph->agraph, input_node, ag_node, NULL, 1);
        if (edge == NULL) {
            return CGRAD_GRAPH_ERR_ALLOC_FAILED;
        }

        // Set slot attribute
        char slot_str[16];
        snprintf(slot_str, sizeof(slot_str), "%d", i);
        agsafeset(edge, "slot", slot_str, "");
        
        // Increment reference count of input node
        cgrad_graph_node* input_node_metadata;
        ret = cgrad_compute_graph_get_node(graph, input_node_ids[i], &input_node_metadata);
        if (ret == CGRAD_SUCCESS) {
            input_node_metadata->ref_count++;
        }
    }

    uuid_copy(out_node_id, node->node_id);
    return CGRAD_SUCCESS;
}

// ============================================================================
// Graph Visualization and Debugging
// ============================================================================

const char* cgrad_op_type_to_string(cgrad_op_type op_type) {
    switch (op_type) {
        case CGRAD_OP_NONE: return "LEAF";
        case CGRAD_OP_AXPY: return "AXPY";
        case CGRAD_OP_GEMM: return "GEMM";
        case CGRAD_OP_TRANSPOSE: return "TRANSPOSE";
        case CGRAD_OP_RESHAPE: return "RESHAPE";
        case CGRAD_OP_REDUCE_SUM: return "REDUCE_SUM";
        default: return "UNKNOWN";
    }
}

int cgrad_compute_graph_to_dot(
    const cgrad_compute_graph* graph,
    const char* filename
) {
    if (graph == NULL || filename == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        return CGRAD_GRAPH_ERR_EXECUTION_FAILED;
    }

    // Use libcgraph's built-in DOT writing
    agwrite(graph->agraph, fp);
    fclose(fp);

    return CGRAD_SUCCESS;
}

void cgrad_compute_graph_print(const cgrad_compute_graph* graph) {
    if (graph == NULL) {
        printf("Graph: NULL\n");
        return;
    }

    char graph_id_str[64];
    uuid_to_string(graph->graph_id, graph_id_str);
    printf("=== Computation Graph ===\n");
    printf("Graph ID: %s\n", graph_id_str);
    printf("Number of nodes: %u\n", HASH_COUNT(graph->node_metadata_table));
    printf("\nNodes:\n");

    cgrad_graph_node *node, *tmp;
    HASH_ITER(hh, graph->node_metadata_table, node, tmp) {
        cgrad_graph_node_print(node);
    }
}

void cgrad_graph_node_print(const cgrad_graph_node* node) {
    if (node == NULL) {
        printf("  Node: NULL\n");
        return;
    }

    char node_id_str[64];
    uuid_to_string(node->node_id, node_id_str);
    
    printf("  Node ID: %s\n", node_id_str);
    printf("    Op: %s\n", cgrad_op_type_to_string(node->op_info.type));
    printf("    Shape: ");
    cgrad_storage_layout_print_shape(&node->layout, TENSOR_DIM);
    printf("    Storage: %s\n", node->storage ? "materialized" : "lazy");
}

// ============================================================================
// Graph Execution
// ============================================================================

/**
 * @brief Execute a single operation node in the graph.
 */
static int execute_node(cgrad_compute_graph* graph, cgrad_graph_node* node) {
    if (node->storage != NULL) {
        return CGRAD_SUCCESS;  // Already computed
    }

    // Get input nodes
    uuid_t input_ids[MAX_NODE_INPUTS];
    int num_inputs;
    int ret = cgrad_compute_graph_get_inputs(graph, node->node_id, input_ids, MAX_NODE_INPUTS, &num_inputs);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    // Get input storages
    cgrad_storage* input_storages[MAX_NODE_INPUTS];
    for (int i = 0; i < num_inputs; i++) {
        cgrad_graph_node* input_node;
        ret = cgrad_compute_graph_get_node(graph, input_ids[i], &input_node);
        if (ret != CGRAD_SUCCESS) {
            return ret;
        }
        if (input_node->storage == NULL) {
            return CGRAD_GRAPH_ERR_EXECUTION_FAILED;  // Input not computed
        }
        input_storages[i] = input_node->storage;
    }

    // Allocate output storage
    cgrad_storage* out_storage = (cgrad_storage*)calloc(1, sizeof(cgrad_storage));
    if (out_storage == NULL) {
        return CGRAD_GRAPH_ERR_ALLOC_FAILED;
    }

    // Get operation descriptor
    const cgrad_op_descriptor* op_desc = cgrad_get_op_descriptor(node->op_info.type);
    if (op_desc == NULL || op_desc->forward == NULL) {
        free(out_storage);
        return CGRAD_GRAPH_ERR_INVALID_OPERATION;
    }

    // Call forward function
    ret = op_desc->forward(
        input_storages,
        num_inputs,
        &node->op_info.metadata,
        out_storage,
        &node->ctx
    );

    if (ret != CGRAD_SUCCESS) {
        cgrad_storage_free(out_storage);
        free(out_storage);
        return ret;
    }

    // Check the backend
    if (out_storage->backend->type != node->backend_type) {
        return CGRAD_GRAPH_ERR_BACKEND_MISMATCH;
    }

    // Cache the result
    node->storage = out_storage;
    return CGRAD_SUCCESS;
}

int cgrad_compute_graph_execute(
    cgrad_compute_graph* graph,
    const uuid_t target_node_id
) {
    if (graph == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    // Perform topological sort to get execution order
    uuid_t sorted_ids[MAX_GRAPH_NODES];
    int num_nodes;
    int ret = cgrad_compute_graph_topological_sort(
        graph, target_node_id,
        sorted_ids, MAX_GRAPH_NODES, &num_nodes
    );
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    // Execute nodes in topological order
    for (int i = 0; i < num_nodes; i++) {
        cgrad_graph_node* node;
        ret = cgrad_compute_graph_get_node(graph, sorted_ids[i], &node);
        if (ret != CGRAD_SUCCESS) {
            return ret;
        }

        // Skip leaf nodes (already have storage)
        if (node->op_info.type == CGRAD_OP_NONE) {
            continue;
        }

        // Execute operation node
        ret = execute_node(graph, node);
        if (ret != CGRAD_SUCCESS) {
            return ret;
        }
    }

    return CGRAD_SUCCESS;
}

// ============================================================================
// Reference Counting Functions
// ============================================================================

int cgrad_compute_graph_increment_ref(cgrad_compute_graph* graph, const uuid_t node_id) {
    if (graph == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_graph_node* node;
    int ret = cgrad_compute_graph_get_node(graph, node_id, &node);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    node->ref_count++;
    return CGRAD_SUCCESS;
}

int cgrad_compute_graph_decrement_ref(cgrad_compute_graph* graph, const uuid_t node_id) {
    if (graph == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    cgrad_graph_node* node;
    int ret = cgrad_compute_graph_get_node(graph, node_id, &node);
    if (ret != CGRAD_SUCCESS) {
        return ret;
    }

    // Decrement reference count
    node->ref_count--;

    // If reference count reaches zero, free the node
    if (node->ref_count == 0) {
        ret = cgrad_compute_graph_free_node(graph, node);
        if (ret != CGRAD_SUCCESS) {
            return ret;
        }
    }

    return CGRAD_SUCCESS;
}

int cgrad_compute_graph_free_node(cgrad_compute_graph* graph, cgrad_graph_node* node) {
    if (graph == NULL || node == NULL) {
        return CGRAD_ERR_NULL_POINTER;
    }

    printf("FREE A\n");

    // Get input nodes before freeing
    uuid_t input_ids[MAX_NODE_INPUTS];
    int num_inputs = 0;
    int ret = cgrad_compute_graph_get_inputs(graph, node->node_id, input_ids, MAX_NODE_INPUTS, &num_inputs);
    
    // Free storage if it exists
    if (node->storage != NULL) {
        cgrad_storage_free(node->storage);
        free(node->storage);
        node->storage = NULL;
    }
    
    // Free gradient storage if it exists
    if (node->grad_storage != NULL) {
        cgrad_storage_free(node->grad_storage);
        free(node->grad_storage);
        node->grad_storage = NULL;
    }
    
    // Context should already be freed by backward, but check just in case
    // Note: We can't free ctx here without knowing its type, so operations
    // must ensure they free their own context in backward()
    node->ctx = NULL;

    // Remove node from metadata table
    HASH_DEL(graph->node_metadata_table, node);
    
    // Remove node from libcgraph
    char node_name[64];
    uuid_to_string(node->node_id, node_name);
    Agnode_t* ag_node = agnode(graph->agraph, node_name, 0);
    if (ag_node != NULL) {
        agdelnode(graph->agraph, ag_node);
    }

    free(node);

    // Recursively decrement ref_count of input nodes
    if (ret == CGRAD_SUCCESS) {
        for (int i = 0; i < num_inputs; i++) {
            cgrad_compute_graph_decrement_ref(graph, input_ids[i]);
        }
    }
    
    printf("FREE B\n");

    return CGRAD_SUCCESS;
}

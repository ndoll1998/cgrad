#include "cgrad_compute_graph.h"
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
        
        free(node);
    }

    // Free libcgraph
    if (graph->agraph != NULL) {
        agclose(graph->agraph);
    }

    return CGRAD_SUCCESS;
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

    uuid_generate(node->node_id);
    node->op_info.type = CGRAD_OP_NONE;
    node->layout = *layout;
    node->storage = storage;
    
    // Track backend type from storage
    if (storage->backend == NULL) {
        free(node);
        return CGRAD_ERR_NULL_POINTER;
    }
    node->backend_type = storage->backend->type;

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
    node->storage = NULL;  // Not computed yet
    node->backend_type = backend_type;  // Inherit backend from inputs

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

    // Add edges from inputs
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
        case CGRAD_OP_ADD: return "ADD";
        case CGRAD_OP_SUB: return "SUB";
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

    // Execute operation based on type
    switch (node->op_info.type) {
        case CGRAD_OP_ADD:
            ret = cgrad_storage_add(input_storages[0], input_storages[1], out_storage);
            break;

        case CGRAD_OP_SUB:
            ret = cgrad_storage_sub(input_storages[0], input_storages[1], out_storage);
            break;

        case CGRAD_OP_GEMM:
            ret = cgrad_storage_gemm(input_storages[0], input_storages[1], out_storage);
            break;

        case CGRAD_OP_TRANSPOSE: {
            ret = cgrad_storage_shallow_copy(input_storages[0], out_storage);
            if (ret != CGRAD_SUCCESS) {
                free(out_storage);
                return ret;
            }
            ret = cgrad_storage_transpose(
                out_storage, 
                node->op_info.metadata.transpose.perm,
                node->op_info.metadata.transpose.ndim
            );
            break;
        }

        case CGRAD_OP_RESHAPE: {
            ret = cgrad_storage_reshape(
                input_storages[0], out_storage,
                node->op_info.metadata.reshape.new_shape,
                node->op_info.metadata.reshape.ndim
            );
            break;
        }

        case CGRAD_OP_REDUCE_SUM:
            ret = cgrad_storage_sum(
                input_storages[0],
                node->op_info.metadata.reduce_sum.mask,
                node->op_info.metadata.reduce_sum.ndim,
                out_storage
            );
            break;

        default:
            free(out_storage);
            return CGRAD_GRAPH_ERR_INVALID_OPERATION;
    }

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

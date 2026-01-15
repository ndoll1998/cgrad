#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>

#include "autograd/cgrad_compute_graph.h"
#include "cgrad_errors.h"

// ============================================================================
// Test: Graph Creation
// ============================================================================

static void test_cgrad_compute_graph_create(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    int ret = cgrad_compute_graph_create(&graph);
    
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_non_null(graph.agraph);
    assert_null(graph.node_metadata_table);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: Add Leaf Node
// ============================================================================

static void test_cgrad_compute_graph_add_leaf_node(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    // Create layout
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 3};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    // Create storage
    cgrad_storage* storage = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    // Add leaf node
    uuid_t node_id;
    int ret = cgrad_compute_graph_add_leaf(&graph, &layout, storage, node_id);
    
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Verify node exists
    cgrad_graph_node* node;
    ret = cgrad_compute_graph_get_node(&graph, node_id, &node);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_non_null(node);
    assert_int_equal(node->op_info.type, CGRAD_OP_NONE);
    // Shape is stored in last dimensions of TENSOR_DIM (8)
    assert_int_equal(node->layout.shape[TENSOR_DIM - 2], 2);
    assert_int_equal(node->layout.shape[TENSOR_DIM - 1], 3);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: Add Operation Node
// ============================================================================

static void test_cgrad_compute_graph_add_op_node(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    // Create two leaf nodes
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 3};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    cgrad_storage* storage1 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage1, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leaf1_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage1, leaf1_id);
    
    cgrad_storage* storage2 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage2, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leaf2_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage2, leaf2_id);
    
    // Add operation node (ADD)
    cgrad_op_info op_info;
    op_info.type = CGRAD_OP_ADD;
    
    uuid_t input_ids[] = {0};
    uuid_copy(input_ids[0], leaf1_id);
    uuid_t input_ids2[2];
    uuid_copy(input_ids2[0], leaf1_id);
    uuid_copy(input_ids2[1], leaf2_id);
    
    uuid_t op_node_id;
    int ret = cgrad_compute_graph_add_op(&graph, &op_info, &layout, input_ids2, 2, op_node_id);
    
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Verify operation node
    cgrad_graph_node* op_node;
    ret = cgrad_compute_graph_get_node(&graph, op_node_id, &op_node);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(op_node->op_info.type, CGRAD_OP_ADD);
    
    // Verify inputs
    uuid_t retrieved_inputs[16];
    int num_inputs;
    ret = cgrad_compute_graph_get_inputs(&graph, op_node_id, retrieved_inputs, 16, &num_inputs);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(num_inputs, 2);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: Topological Sort
// ============================================================================

static void test_cgrad_compute_graph_topological_sort(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    // Create a simple chain: A -> ADD -> B
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 3};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    // Leaf A
    cgrad_storage* storageA = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storageA, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leafA_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storageA, leafA_id);
    
    // Leaf B
    cgrad_storage* storageB = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storageB, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leafB_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storageB, leafB_id);
    
    // ADD node
    cgrad_op_info op_info;
    op_info.type = CGRAD_OP_ADD;
    uuid_t input_ids[2];
    uuid_copy(input_ids[0], leafA_id);
    uuid_copy(input_ids[1], leafB_id);
    uuid_t add_id;
    cgrad_compute_graph_add_op(&graph, &op_info, &layout, input_ids, 2, add_id);
    
    // Topological sort from ADD node
    uuid_t sorted[256];
    int num_sorted;
    int ret = cgrad_compute_graph_topological_sort(&graph, add_id, sorted, 256, &num_sorted);
    
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(num_sorted, 3);  // A, B, ADD
    
    // Last node should be the ADD node
    assert_true(uuid_compare(sorted[num_sorted - 1], add_id) == 0);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: DOT Export
// ============================================================================

static void test_cgrad_compute_graph_dot_export(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    // Create simple graph
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 2};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    cgrad_storage* storage = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leaf_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage, leaf_id);
    
    // Export to DOT
    int ret = cgrad_compute_graph_to_dot(&graph, "/tmp/test_graph.dot");
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: Backend Type Tracking
// ============================================================================

static void test_cgrad_compute_graph_backend_type_tracking(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    // Create layout
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 3};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    // Create leaf node with F32_CPU backend
    cgrad_storage* storage = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leaf_id;
    int ret = cgrad_compute_graph_add_leaf(&graph, &layout, storage, leaf_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Verify backend type is tracked
    cgrad_graph_node* node;
    ret = cgrad_compute_graph_get_node(&graph, leaf_id, &node);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(node->backend_type, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: Backend Consistency for Operations
// ============================================================================

static void test_cgrad_compute_graph_backend_consistency_same_backend(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    // Create two leaf nodes with same backend
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 3};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    cgrad_storage* storage1 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage1, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leaf1_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage1, leaf1_id);
    
    cgrad_storage* storage2 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage2, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leaf2_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage2, leaf2_id);
    
    // Add operation node - should succeed with same backend
    cgrad_op_info op_info;
    op_info.type = CGRAD_OP_ADD;
    uuid_t input_ids[2];
    uuid_copy(input_ids[0], leaf1_id);
    uuid_copy(input_ids[1], leaf2_id);
    
    uuid_t op_node_id;
    int ret = cgrad_compute_graph_add_op(&graph, &op_info, &layout, input_ids, 2, op_node_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Verify operation node inherits backend type
    cgrad_graph_node* op_node;
    ret = cgrad_compute_graph_get_node(&graph, op_node_id, &op_node);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(op_node->backend_type, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    cgrad_compute_graph_free(&graph);
}

// NOTE: Test for mixed backends (CGRAD_GRAPH_ERR_BACKEND_MISMATCH) will be added
// once additional backend types (e.g., GPU, CUDA) are implemented in the system.

// ============================================================================
// Test: Reference Counting - Leaf Node
// ============================================================================

static void test_cgrad_compute_graph_refcount_leaf_node(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    // Create layout
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 3};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    // Create storage
    cgrad_storage* storage = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    // Add leaf node
    uuid_t node_id;
    int ret = cgrad_compute_graph_add_leaf(&graph, &layout, storage, node_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Check node count
    assert_int_equal(cgrad_compute_graph_get_node_count(&graph), 1);
    
    // Check initial ref_count
    cgrad_graph_node* node;
    ret = cgrad_compute_graph_get_node(&graph, node_id, &node);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(node->ref_count, 1);
    
    // Decrement ref_count (simulating tensor free)
    ret = cgrad_compute_graph_decrement_ref(&graph, node_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Node should be freed (can't access it anymore)
    ret = cgrad_compute_graph_get_node(&graph, node_id, &node);
    assert_int_equal(ret, CGRAD_GRAPH_ERR_NODE_NOT_FOUND);
    
    // Check node count is now 0
    assert_int_equal(cgrad_compute_graph_get_node_count(&graph), 0);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: Reference Counting - Increment/Decrement
// ============================================================================

static void test_cgrad_compute_graph_refcount_increment_decrement(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    // Create layout
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 3};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    // Create storage
    cgrad_storage* storage = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    // Add leaf node
    uuid_t node_id;
    int ret = cgrad_compute_graph_add_leaf(&graph, &layout, storage, node_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Get node and check initial ref_count
    cgrad_graph_node* node;
    ret = cgrad_compute_graph_get_node(&graph, node_id, &node);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(node->ref_count, 1);
    
    // Increment ref_count (simulating tensor copy)
    ret = cgrad_compute_graph_increment_ref(&graph, node_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(node->ref_count, 2);
    
    // Decrement once
    ret = cgrad_compute_graph_decrement_ref(&graph, node_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Node should still exist
    ret = cgrad_compute_graph_get_node(&graph, node_id, &node);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(node->ref_count, 1);
    
    // Decrement again
    ret = cgrad_compute_graph_decrement_ref(&graph, node_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Node should be freed
    ret = cgrad_compute_graph_get_node(&graph, node_id, &node);
    assert_int_equal(ret, CGRAD_GRAPH_ERR_NODE_NOT_FOUND);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: Reference Counting - Operation Nodes
// ============================================================================

static void test_cgrad_compute_graph_refcount_operation_nodes(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    // Create two leaf nodes
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 3};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    cgrad_storage* storage1 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage1, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leaf1_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage1, leaf1_id);
    
    cgrad_storage* storage2 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage2, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leaf2_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage2, leaf2_id);
    
    // Check initial ref_counts
    cgrad_graph_node* node1;
    cgrad_graph_node* node2;
    cgrad_compute_graph_get_node(&graph, leaf1_id, &node1);
    cgrad_compute_graph_get_node(&graph, leaf2_id, &node2);
    assert_int_equal(node1->ref_count, 1);
    assert_int_equal(node2->ref_count, 1);
    
    // Add operation node (ADD)
    cgrad_op_info op_info;
    op_info.type = CGRAD_OP_ADD;
    uuid_t input_ids[2];
    uuid_copy(input_ids[0], leaf1_id);
    uuid_copy(input_ids[1], leaf2_id);
    
    uuid_t op_node_id;
    int ret = cgrad_compute_graph_add_op(&graph, &op_info, &layout, input_ids, 2, op_node_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Check ref_counts increased (inputs are referenced by operation)
    assert_int_equal(node1->ref_count, 2);
    assert_int_equal(node2->ref_count, 2);
    
    // Get operation node
    cgrad_graph_node* op_node;
    ret = cgrad_compute_graph_get_node(&graph, op_node_id, &op_node);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(op_node->ref_count, 1);
    
    // Check node count (2 leaves + 1 op = 3)
    assert_int_equal(cgrad_compute_graph_get_node_count(&graph), 3);
    
    // Free operation node
    ret = cgrad_compute_graph_decrement_ref(&graph, op_node_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Check ref_counts decreased (cascading decrement)
    assert_int_equal(node1->ref_count, 1);
    assert_int_equal(node2->ref_count, 1);
    
    // Check node count (op node freed, 2 leaves remain)
    assert_int_equal(cgrad_compute_graph_get_node_count(&graph), 2);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: Reference Counting - Shared Subgraph
// ============================================================================

static void test_cgrad_compute_graph_refcount_shared_subgraph(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    // Create two leaf nodes
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 3};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    cgrad_storage* storage1 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage1, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leaf1_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage1, leaf1_id);
    
    cgrad_storage* storage2 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage2, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leaf2_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage2, leaf2_id);
    
    // c = a + b
    cgrad_op_info op_info;
    op_info.type = CGRAD_OP_ADD;
    uuid_t input_ids[2];
    uuid_copy(input_ids[0], leaf1_id);
    uuid_copy(input_ids[1], leaf2_id);
    uuid_t c_id;
    cgrad_compute_graph_add_op(&graph, &op_info, &layout, input_ids, 2, c_id);
    
    // d = c + c (c is used twice)
    uuid_copy(input_ids[0], c_id);
    uuid_copy(input_ids[1], c_id);
    uuid_t d_id;
    cgrad_compute_graph_add_op(&graph, &op_info, &layout, input_ids, 2, d_id);
    
    // Check c's ref_count (1 initial + 2 from d)
    cgrad_graph_node* c_node;
    cgrad_compute_graph_get_node(&graph, c_id, &c_node);
    assert_int_equal(c_node->ref_count, 3);
    
    // Decrement c once (simulating tensor free)
    int ret = cgrad_compute_graph_decrement_ref(&graph, c_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // c should still exist (ref_count = 2)
    ret = cgrad_compute_graph_get_node(&graph, c_id, &c_node);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(c_node->ref_count, 2);
    
    // Free d
    ret = cgrad_compute_graph_decrement_ref(&graph, d_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Now c should be freed (cascading cleanup)
    ret = cgrad_compute_graph_get_node(&graph, c_id, &c_node);
    assert_int_equal(ret, CGRAD_GRAPH_ERR_NODE_NOT_FOUND);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: Reference Counting - Complex Graph
// ============================================================================

static void test_cgrad_compute_graph_refcount_complex_graph(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    // Create two leaf nodes
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 3};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    cgrad_storage* storage1 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage1, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t a_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage1, a_id);
    
    cgrad_storage* storage2 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage2, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t b_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage2, b_id);
    
    // c = a + b
    cgrad_op_info add_op;
    add_op.type = CGRAD_OP_ADD;
    uuid_t input_ids[2];
    uuid_copy(input_ids[0], a_id);
    uuid_copy(input_ids[1], b_id);
    uuid_t c_id;
    cgrad_compute_graph_add_op(&graph, &add_op, &layout, input_ids, 2, c_id);
    
    // d = a - b
    cgrad_op_info sub_op;
    sub_op.type = CGRAD_OP_SUB;
    uuid_copy(input_ids[0], a_id);
    uuid_copy(input_ids[1], b_id);
    uuid_t d_id;
    cgrad_compute_graph_add_op(&graph, &sub_op, &layout, input_ids, 2, d_id);
    
    // e = c + d
    uuid_copy(input_ids[0], c_id);
    uuid_copy(input_ids[1], d_id);
    uuid_t e_id;
    cgrad_compute_graph_add_op(&graph, &add_op, &layout, input_ids, 2, e_id);
    
    // Check ref_counts
    cgrad_graph_node *a_node, *b_node, *c_node, *d_node;
    cgrad_compute_graph_get_node(&graph, a_id, &a_node);
    cgrad_compute_graph_get_node(&graph, b_id, &b_node);
    cgrad_compute_graph_get_node(&graph, c_id, &c_node);
    cgrad_compute_graph_get_node(&graph, d_id, &d_node);
    
    // a and b are referenced by c and d (1 + 2 = 3)
    assert_int_equal(a_node->ref_count, 3);
    assert_int_equal(b_node->ref_count, 3);
    // c and d are referenced by e (1 + 1 = 2)
    assert_int_equal(c_node->ref_count, 2);
    assert_int_equal(d_node->ref_count, 2);
    
    // Free in arbitrary order: c, e, d, a, b
    int ret = cgrad_compute_graph_decrement_ref(&graph, c_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // c should still exist (ref_count = 1, referenced by e)
    ret = cgrad_compute_graph_get_node(&graph, c_id, &c_node);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(c_node->ref_count, 1);
    
    // Free e (triggers cascading cleanup of c and d)
    ret = cgrad_compute_graph_decrement_ref(&graph, e_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // c should be freed (cascading cleanup from e)
    ret = cgrad_compute_graph_get_node(&graph, c_id, &c_node);
    assert_int_equal(ret, CGRAD_GRAPH_ERR_NODE_NOT_FOUND);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: Backward - requires_grad default
// ============================================================================

static void test_cgrad_compute_graph_backward_requires_grad_default(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 3};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    cgrad_storage* storage = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    uuid_t node_id;
    int ret = cgrad_compute_graph_add_leaf(&graph, &layout, storage, node_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Default should be requires_grad = 1
    cgrad_graph_node* node;
    ret = cgrad_compute_graph_get_node(&graph, node_id, &node);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(node->requires_grad, 1);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: Backward - requires_grad set
// ============================================================================

static void test_cgrad_compute_graph_backward_requires_grad_set(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 3};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    cgrad_storage* storage = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    uuid_t node_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage, node_id);
    
    // Set requires_grad to false
    int ret = cgrad_compute_graph_set_requires_grad(&graph, node_id, 0);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_graph_node* node;
    ret = cgrad_compute_graph_get_node(&graph, node_id, &node);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(node->requires_grad, 0);
    
    // Set back to true
    ret = cgrad_compute_graph_set_requires_grad(&graph, node_id, 1);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(node->requires_grad, 1);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: Backward - requires_grad inheritance
// ============================================================================

static void test_cgrad_compute_graph_backward_requires_grad_inheritance(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 3};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    // Create two leaf nodes
    cgrad_storage* storage1 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage1, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leaf1_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage1, leaf1_id);
    
    cgrad_storage* storage2 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage2, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leaf2_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage2, leaf2_id);
    
    // Set leaf1 to not require grad
    cgrad_compute_graph_set_requires_grad(&graph, leaf1_id, 0);
    
    // Add operation - should inherit requires_grad from leaf2
    cgrad_op_info op_info;
    op_info.type = CGRAD_OP_ADD;
    uuid_t input_ids[2];
    uuid_copy(input_ids[0], leaf1_id);
    uuid_copy(input_ids[1], leaf2_id);
    
    uuid_t op_node_id;
    int ret = cgrad_compute_graph_add_op(&graph, &op_info, &layout, input_ids, 2, op_node_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_graph_node* op_node;
    ret = cgrad_compute_graph_get_node(&graph, op_node_id, &op_node);
    assert_int_equal(ret, CGRAD_SUCCESS);
    assert_int_equal(op_node->requires_grad, 1);  // Inherited from leaf2
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: Backward - requires forward execution
// ============================================================================

static void test_cgrad_compute_graph_backward_requires_forward(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 2};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    cgrad_storage* storage1 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage1, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leaf1_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage1, leaf1_id);
    
    cgrad_storage* storage2 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage2, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    uuid_t leaf2_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage2, leaf2_id);
    
    cgrad_op_info op_info;
    op_info.type = CGRAD_OP_ADD;
    uuid_t input_ids[2];
    uuid_copy(input_ids[0], leaf1_id);
    uuid_copy(input_ids[1], leaf2_id);
    
    uuid_t op_node_id;
    cgrad_compute_graph_add_op(&graph, &op_info, &layout, input_ids, 2, op_node_id);
    
    // Backward without forward should fail
    int ret = cgrad_compute_graph_backward(&graph, op_node_id);
    assert_int_equal(ret, CGRAD_GRAPH_ERR_FORWARD_NOT_EXECUTED);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: Backward - grad_storage initialization
// ============================================================================

static void test_cgrad_compute_graph_backward_grad_storage_init(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 2};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    cgrad_storage* storage = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    
    uuid_t node_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage, node_id);
    
    // Initially grad_storage should be NULL
    cgrad_graph_node* node;
    cgrad_compute_graph_get_node(&graph, node_id, &node);
    assert_null(node->grad_storage);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test: Backward - zero_grad
// ============================================================================

static void test_cgrad_compute_graph_backward_zero_grad(void **state) {
    (void) state;
    
    cgrad_compute_graph graph;
    cgrad_compute_graph_create(&graph);
    
    cgrad_storage_layout layout;
    uint32_t shape[] = {2, 2};
    cgrad_storage_layout_init(&layout, shape, 2);
    
    cgrad_storage* storage1 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage1, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(storage1, 1.0f);
    uuid_t leaf1_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage1, leaf1_id);
    
    cgrad_storage* storage2 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage_init(storage2, shape, 2, CGRAD_STORAGE_BACKEND_F32_CPU);
    cgrad_storage_fill(storage2, 2.0f);
    uuid_t leaf2_id;
    cgrad_compute_graph_add_leaf(&graph, &layout, storage2, leaf2_id);
    
    cgrad_op_info op_info;
    op_info.type = CGRAD_OP_ADD;
    uuid_t input_ids[2];
    uuid_copy(input_ids[0], leaf1_id);
    uuid_copy(input_ids[1], leaf2_id);
    
    uuid_t op_node_id;
    cgrad_compute_graph_add_op(&graph, &op_info, &layout, input_ids, 2, op_node_id);
    
    // Execute forward
    int ret = cgrad_compute_graph_execute(&graph, op_node_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Execute backward
    ret = cgrad_compute_graph_backward(&graph, op_node_id);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    // Gradients should exist
    cgrad_graph_node* leaf1;
    cgrad_compute_graph_get_node(&graph, leaf1_id, &leaf1);
    assert_non_null(leaf1->grad_storage);
    
    // Zero gradients
    ret = cgrad_compute_graph_zero_grad(&graph);
    assert_int_equal(ret, CGRAD_SUCCESS);
    
    cgrad_compute_graph_free(&graph);
}

// ============================================================================
// Test Suite
// ============================================================================

int run_cgrad_compute_graph_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cgrad_compute_graph_create),
        cmocka_unit_test(test_cgrad_compute_graph_add_leaf_node),
        cmocka_unit_test(test_cgrad_compute_graph_add_op_node),
        cmocka_unit_test(test_cgrad_compute_graph_topological_sort),
        cmocka_unit_test(test_cgrad_compute_graph_dot_export),
        cmocka_unit_test(test_cgrad_compute_graph_backend_type_tracking),
        cmocka_unit_test(test_cgrad_compute_graph_backend_consistency_same_backend),
        cmocka_unit_test(test_cgrad_compute_graph_refcount_leaf_node),
        cmocka_unit_test(test_cgrad_compute_graph_refcount_increment_decrement),
        cmocka_unit_test(test_cgrad_compute_graph_refcount_operation_nodes),
        cmocka_unit_test(test_cgrad_compute_graph_refcount_shared_subgraph),
        cmocka_unit_test(test_cgrad_compute_graph_refcount_complex_graph),
        cmocka_unit_test(test_cgrad_compute_graph_backward_requires_grad_default),
        cmocka_unit_test(test_cgrad_compute_graph_backward_requires_grad_set),
        cmocka_unit_test(test_cgrad_compute_graph_backward_requires_grad_inheritance),
        cmocka_unit_test(test_cgrad_compute_graph_backward_requires_forward),
        cmocka_unit_test(test_cgrad_compute_graph_backward_grad_storage_init),
        cmocka_unit_test(test_cgrad_compute_graph_backward_zero_grad),
    };
    
    return cmocka_run_group_tests_name("cgrad_compute_graph", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_compute_graph_tests();
}
#endif

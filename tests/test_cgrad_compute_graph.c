#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>

#include "cgrad_compute_graph.h"
#include "cgrad_errors.h"

// ============================================================================
// Test: Graph Creation
// ============================================================================

static void test_graph_create(void **state) {
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

static void test_add_leaf_node(void **state) {
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

static void test_add_op_node(void **state) {
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

static void test_topological_sort(void **state) {
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

static void test_dot_export(void **state) {
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

static void test_backend_type_tracking(void **state) {
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

static void test_backend_consistency_same_backend(void **state) {
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
// Test Suite
// ============================================================================

#ifndef TEST_ALL_MAIN
int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_graph_create),
        cmocka_unit_test(test_add_leaf_node),
        cmocka_unit_test(test_add_op_node),
        cmocka_unit_test(test_topological_sort),
        cmocka_unit_test(test_dot_export),
        cmocka_unit_test(test_backend_type_tracking),
        cmocka_unit_test(test_backend_consistency_same_backend),
    };
    
    return cmocka_run_group_tests(tests, NULL, NULL);
}
#else
int test_cgrad_compute_graph_main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_graph_create),
        cmocka_unit_test(test_add_leaf_node),
        cmocka_unit_test(test_add_op_node),
        cmocka_unit_test(test_topological_sort),
        cmocka_unit_test(test_dot_export),
        cmocka_unit_test(test_backend_type_tracking),
        cmocka_unit_test(test_backend_consistency_same_backend),
    };
    
    return _cmocka_run_group_tests("cgrad_compute_graph", tests, sizeof(tests)/sizeof(tests[0]), NULL, NULL);
}
#endif

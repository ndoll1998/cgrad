#include <cmocka.h>
#include "storage/cgrad_storage_registry.h"
#include "storage/cgrad_storage.h"
#include "cgrad_status.h"
#include <stdlib.h>

// ============================================================================
// Setup and Teardown
// ============================================================================

static int global_registry_setup_test(void **state) {
    // Initialize the global registry for testing
    int rc = cgrad_storage_registry_init();
    assert_int_equal(rc, CGRAD_SUCCESS);
    
    return 0;
}

static int global_registry_teardown_test(void **state) {
    // Free the global registry
    cgrad_storage_registry_free();
    return 0;
}

static void test_cgrad_storage_register_root_and_find(void **state) {
    cgrad_storage* tensor = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    assert_non_null(tensor);
    uuid_generate(tensor->uuid);

    int rc = cgrad_storage_registry_register(tensor, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);

    cgrad_storage root;
    rc = cgrad_storage_registry_get_root(tensor, &root);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_ptr_equal(root.data, tensor->data);

    // Deregister and check registry state
    rc = cgrad_storage_registry_deregister(tensor);
    assert_int_equal(rc, CGRAD_SUCCESS);

    // Deregister again (should return error)
    rc = cgrad_storage_registry_deregister(tensor);
    assert_int_equal(rc, CGRAD_ERR_STORAGE_REGISTRY_PARENT_NOT_REGISTERED);

    free(tensor);
}

static void test_cgrad_storage_register_child_and_bucket_sharing(void **state) {
    cgrad_storage* root = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage* child = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    assert_non_null(root);
    assert_non_null(child);
    uuid_generate(root->uuid);
    uuid_generate(child->uuid);

    int rc = cgrad_storage_registry_register(root, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);

    rc = cgrad_storage_registry_register(child, root);
    assert_int_equal(rc, CGRAD_SUCCESS);

    cgrad_storage root1, root2;
    rc = cgrad_storage_registry_get_root(root, &root1);
    assert_int_equal(rc, CGRAD_SUCCESS);
    rc = cgrad_storage_registry_get_root(child, &root2);
    assert_int_equal(rc, CGRAD_SUCCESS);

    // Both should have the same root
    assert_int_equal(uuid_compare(root1.uuid, root2.uuid), 0);
    assert_ptr_equal(root1.data, root->data);
    assert_ptr_equal(root2.data, root->data);

    // Check bucket size
    assert_int_equal(cgrad_storage_registry_bucket_get_size(root), 2);

    // Deregister child, bucket should not be empty
    rc = cgrad_storage_registry_deregister(child);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_registry_bucket_get_size(root), 1);

    // Deregister root, now bucket should be empty
    rc = cgrad_storage_registry_deregister(root);
    assert_int_equal(rc, CGRAD_SUCCESS);

    free(child);
    free(root);
}

static void test_cgrad_storage_register_with_unregistered_parent(void **state) {
    cgrad_storage* parent = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage* child = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    assert_non_null(parent);
    assert_non_null(child);
    uuid_generate(parent->uuid);
    uuid_generate(child->uuid);

    // Do not register parent
    int rc = cgrad_storage_registry_register(child, parent);
    assert_int_equal(rc, CGRAD_ERR_STORAGE_REGISTRY_PARENT_NOT_REGISTERED);

    free(child);
    free(parent);
}

static void test_cgrad_storage_register_count(void **state) {
    // Should be empty at start
    assert_int_equal(cgrad_storage_registry_count(), 0);

    cgrad_storage* t1 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage* t2 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    assert_non_null(t1);
    assert_non_null(t2);
    uuid_generate(t1->uuid);
    uuid_generate(t2->uuid);

    // Register t1
    int rc = cgrad_storage_registry_register(t1, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_registry_count(), 1);

    // Register t2
    rc = cgrad_storage_registry_register(t2, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_registry_count(), 2);

    // Deregister t1
    rc = cgrad_storage_registry_deregister(t1);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_registry_count(), 1);

    // Deregister t2
    rc = cgrad_storage_registry_deregister(t2);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_registry_count(), 0);

    free(t1);
    free(t2);
}

static void test_cgrad_storage_register_idempotency(void **state) {
    cgrad_storage* tensor = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    assert_non_null(tensor);
    uuid_generate(tensor->uuid);

    // Register twice as root
    int rc = cgrad_storage_registry_register(tensor, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);
    rc = cgrad_storage_registry_register(tensor, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);

    // Should only count once
    assert_int_equal(cgrad_storage_registry_count(), 1);

    // Deregister twice
    rc = cgrad_storage_registry_deregister(tensor);
    assert_int_equal(rc, CGRAD_SUCCESS);

    rc = cgrad_storage_registry_deregister(tensor);
    assert_int_equal(rc, CGRAD_ERR_STORAGE_REGISTRY_PARENT_NOT_REGISTERED);

    free(tensor);
}

static void test_cgrad_storage_registry_tracker_basic(void **state) {
    // Start tracking
    cgrad_storage_registry_record* record = cgrad_storage_registry_start_recording();
    assert_non_null(record);
    assert_int_equal(cgrad_storage_registry_record_count(record), 0);
    
    // Register some storages
    cgrad_storage s1 = {0}, s2 = {0}, s3 = {0};
    uuid_generate(s1.uuid);
    uuid_generate(s2.uuid);
    uuid_generate(s3.uuid);
    
    cgrad_storage_registry_register(&s1, NULL);
    cgrad_storage_registry_register(&s2, NULL);
    cgrad_storage_registry_register(&s3, NULL);
    
    // Check tracker captured them
    assert_int_equal(cgrad_storage_registry_record_count(record), 3);
    
    // Deregister s1 while tracker is active - should be removed from tracker
    cgrad_storage_registry_deregister(&s1);
    assert_int_equal(cgrad_storage_registry_record_count(record), 2); // Now 2
    
    // Stop tracking
    assert_int_equal(cgrad_storage_registry_stop_recording(record), CGRAD_SUCCESS);
    
    // Register another storage - should not be tracked
    cgrad_storage s4 = {0};
    uuid_generate(s4.uuid);
    cgrad_storage_registry_register(&s4, NULL);
    
    // Deregister after stopping - should NOT be removed from tracker (tracker is stopped)
    cgrad_storage_registry_deregister(&s2);
    cgrad_storage_registry_deregister(&s3);
    cgrad_storage_registry_deregister(&s4);
}

static void test_cgrad_storage_registry_tracker_nested(void **state) {
    // Start first tracker
    cgrad_storage_registry_record* record1 = cgrad_storage_registry_start_recording();
    assert_non_null(record1);
    
    // Register some storages
    cgrad_storage s1 = {0}, s2 = {0};
    uuid_generate(s1.uuid);
    uuid_generate(s2.uuid);
    cgrad_storage_registry_register(&s1, NULL);
    cgrad_storage_registry_register(&s2, NULL);
    
    // Start nested record
    cgrad_storage_registry_record* record2 = cgrad_storage_registry_start_recording();
    assert_non_null(record2);
    
    // Register more storages
    cgrad_storage s3 = {0}, s4 = {0};
    uuid_generate(s3.uuid);
    uuid_generate(s4.uuid);
    cgrad_storage_registry_register(&s3, NULL);
    cgrad_storage_registry_register(&s4, NULL);
    
    // Check record1 has all 4
    assert_int_equal(cgrad_storage_registry_record_count(record1), 4);
    
    // Check record2 has only 2
    assert_int_equal(cgrad_storage_registry_record_count(record2), 2);
    
    // Deregister s1 while both records are active - should be removed from record1
    cgrad_storage_registry_deregister(&s1);
    assert_int_equal(cgrad_storage_registry_record_count(record1), 3); // Now has s2, s3, s4
    assert_int_equal(cgrad_storage_registry_record_count(record2), 2); // Still has s3, s4
    
    // Stop both records
    assert_int_equal(cgrad_storage_registry_stop_recording(record2), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_registry_stop_recording(record1), CGRAD_SUCCESS);
    
    // Deregister after stopping - records keep their snapshots
    cgrad_storage_registry_deregister(&s2);
    cgrad_storage_registry_deregister(&s3);
    cgrad_storage_registry_deregister(&s4);
}

int run_cgrad_storage_registry_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_cgrad_storage_register_root_and_find, global_registry_setup_test, global_registry_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_register_child_and_bucket_sharing, global_registry_setup_test, global_registry_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_register_with_unregistered_parent, global_registry_setup_test, global_registry_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_register_count, global_registry_setup_test, global_registry_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_register_idempotency, global_registry_setup_test, global_registry_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_registry_tracker_basic, global_registry_setup_test, global_registry_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_registry_tracker_nested, global_registry_setup_test, global_registry_teardown_test),
    };
    return cmocka_run_group_tests_name("cgrad_storage_registry", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_storage_registry_tests();
}
#endif
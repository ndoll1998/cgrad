#include <cmocka.h>
#include "storage/cgrad_storage_registry.h"
#include "storage/cgrad_storage.h"
#include "cgrad_errors.h"
#include <stdlib.h>

// Forward declare getter - it's internal to storage module
// Use the public getter function instead of extern
#include "cgrad.h"

// ============================================================================
// Setup and Teardown
// ============================================================================

static int registry_setup_test(void **state) {
    (void) state;
    cgrad_init();
    return 0;
}

static int registry_teardown_test(void **state) {
    (void) state;
    cgrad_cleanup();
    return 0;
}

static void test_cgrad_storage_register_root_and_find(void **state) {
    (void)state;
    cgrad_storage_registry* registry = cgrad_storage_get_global_registry();
    assert_non_null(registry);
    
    cgrad_storage* tensor = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    assert_non_null(tensor);
    uuid_generate(tensor->uuid);

    int rc = cgrad_storage_registry_register(registry, tensor, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);

    cgrad_storage_registry_entry* reg_entry = NULL;
    HASH_FIND(hh, registry->storage_map, tensor->uuid, sizeof(uuid_t), reg_entry);
    assert_non_null(reg_entry);
    cgrad_storage_registry_bucket* bucket = reg_entry->bucket;
    assert_non_null(bucket);
    assert_ptr_equal(bucket->root.data, tensor->data);

    // Check tensor is in bucket's tensor_map
    cgrad_storage_registry_node* entry = NULL;
    HASH_FIND(hh, bucket->storage_map, tensor->uuid, sizeof(uuid_t), entry);
    assert_non_null(entry);
    assert_ptr_equal(entry->storage, tensor);

    // Deregister and check registry state
    rc = cgrad_storage_registry_deregister(registry, tensor);
    assert_int_equal(rc, CGRAD_SUCCESS);

    // Deregister again (should return error)
    rc = cgrad_storage_registry_deregister(registry, tensor);
    assert_int_equal(rc, CGRAD_STORAGE_REGISTRY_PARENT_NOT_REGISTERED);

    free(tensor);
}

static void test_cgrad_storage_register_child_and_bucket_sharing(void **state) {
    (void)state;
    cgrad_storage_registry* registry = cgrad_storage_get_global_registry();
    assert_non_null(registry);
    
    cgrad_storage* root = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage* child = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    assert_non_null(root);
    assert_non_null(child);
    uuid_generate(root->uuid);
    uuid_generate(child->uuid);

    int rc = cgrad_storage_registry_register(registry, root, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);

    rc = cgrad_storage_registry_register(registry, child, root);
    assert_int_equal(rc, CGRAD_SUCCESS);

    cgrad_storage_registry_entry* root_entry = NULL;
    cgrad_storage_registry_entry* child_entry = NULL;
    HASH_FIND(hh, registry->storage_map, root->uuid, sizeof(uuid_t), root_entry);
    HASH_FIND(hh, registry->storage_map, child->uuid, sizeof(uuid_t), child_entry);

    assert_non_null(root_entry);
    assert_non_null(child_entry);
    cgrad_storage_registry_bucket* root_bucket = root_entry->bucket;
    cgrad_storage_registry_bucket* child_bucket = child_entry->bucket;
    assert_non_null(root_bucket);
    assert_non_null(child_bucket);
    assert_ptr_equal(root_bucket, child_bucket);
    assert_ptr_equal(root_bucket->root.data, root->data);

    // Both tensors should be in the bucket's tensor_map
    cgrad_storage_registry_node* entry = NULL;
    HASH_FIND(hh, root_bucket->storage_map, root->uuid, sizeof(uuid_t), entry);
    assert_non_null(entry);
    HASH_FIND(hh, root_bucket->storage_map, child->uuid, sizeof(uuid_t), entry);
    assert_non_null(entry);

    // Deregister child, bucket should not be empty
    rc = cgrad_storage_registry_deregister(registry, child);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_registry_bucket_get_size(registry, root), 1);

    // Deregister root, now bucket should be empty
    rc = cgrad_storage_registry_deregister(registry, root);
    assert_int_equal(rc, CGRAD_SUCCESS);

    free(child);
    free(root);
}

static void test_cgrad_storage_register_with_unregistered_parent(void **state) {
    (void)state;
    cgrad_storage_registry* registry = cgrad_storage_get_global_registry();
    assert_non_null(registry);
    
    cgrad_storage* parent = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage* child = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    assert_non_null(parent);
    assert_non_null(child);
    uuid_generate(parent->uuid);
    uuid_generate(child->uuid);

    // Do not register parent
    int rc = cgrad_storage_registry_register(registry, child, parent);
    assert_int_equal(rc, CGRAD_STORAGE_REGISTRY_PARENT_NOT_REGISTERED);

    free(child);
    free(parent);
}

static void test_cgrad_storage_register_count(void **state) {
    (void)state;
    cgrad_storage_registry* registry = cgrad_storage_get_global_registry();
    assert_non_null(registry);
    
    // Should be empty at start
    assert_int_equal(cgrad_storage_registry_count(registry), 0);

    cgrad_storage* t1 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    cgrad_storage* t2 = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    assert_non_null(t1);
    assert_non_null(t2);
    uuid_generate(t1->uuid);
    uuid_generate(t2->uuid);

    // Register t1
    int rc = cgrad_storage_registry_register(registry, t1, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_registry_count(registry), 1);

    // Register t2
    rc = cgrad_storage_registry_register(registry, t2, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_registry_count(registry), 2);

    // Deregister t1
    rc = cgrad_storage_registry_deregister(registry, t1);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_registry_count(registry), 1);

    // Deregister t2
    rc = cgrad_storage_registry_deregister(registry, t2);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_registry_count(registry), 0);

    free(t1);
    free(t2);
}

static void test_cgrad_storage_register_idempotency(void **state) {
    (void)state;
    cgrad_storage_registry* registry = cgrad_storage_get_global_registry();
    assert_non_null(registry);
    
    cgrad_storage* tensor = (cgrad_storage*)malloc(sizeof(cgrad_storage));
    assert_non_null(tensor);
    uuid_generate(tensor->uuid);

    // Register twice as root
    int rc = cgrad_storage_registry_register(registry, tensor, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);
    rc = cgrad_storage_registry_register(registry, tensor, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);

    cgrad_storage_registry_entry* reg_entry = NULL;
    HASH_FIND(hh, registry->storage_map, tensor->uuid, sizeof(uuid_t), reg_entry);
    assert_non_null(reg_entry);

    // Deregister twice
    rc = cgrad_storage_registry_deregister(registry, tensor);
    assert_int_equal(rc, CGRAD_SUCCESS);

    rc = cgrad_storage_registry_deregister(registry, tensor);
    assert_int_equal(rc, CGRAD_STORAGE_REGISTRY_PARENT_NOT_REGISTERED);

    free(tensor);
}

static void test_cgrad_storage_registry_tracker_basic(void **state) {
    (void)state;
    cgrad_storage_registry* registry = cgrad_storage_get_global_registry();
    assert_non_null(registry);
    
    // Start tracking
    cgrad_storage_registry_record* record = cgrad_storage_registry_start_recording(registry);
    assert_non_null(record);
    assert_int_equal(cgrad_storage_registry_record_count(record), 0);
    
    // Register some storages
    cgrad_storage s1 = {0}, s2 = {0}, s3 = {0};
    uuid_generate(s1.uuid);
    uuid_generate(s2.uuid);
    uuid_generate(s3.uuid);
    
    cgrad_storage_registry_register(registry, &s1, NULL);
    cgrad_storage_registry_register(registry, &s2, NULL);
    cgrad_storage_registry_register(registry, &s3, NULL);
    
    // Check tracker captured them
    assert_int_equal(cgrad_storage_registry_record_count(record), 3);
    
    // Verify storages are in tracker's hashmap
    cgrad_storage_registry_node* entry = NULL;
    HASH_FIND(hh, record->storage_map, s1.uuid, sizeof(uuid_t), entry);
    assert_non_null(entry);
    HASH_FIND(hh, record->storage_map, s2.uuid, sizeof(uuid_t), entry);
    assert_non_null(entry);
    HASH_FIND(hh, record->storage_map, s3.uuid, sizeof(uuid_t), entry);
    assert_non_null(entry);
    
    // Deregister s1 while tracker is active - should be removed from tracker
    cgrad_storage_registry_deregister(registry, &s1);
    assert_int_equal(cgrad_storage_registry_record_count(record), 2); // Now 2
    
    // Stop tracking
    assert_int_equal(cgrad_storage_registry_stop_recording(registry, record), CGRAD_SUCCESS);
    
    // Register another storage - should not be tracked
    cgrad_storage s4 = {0};
    uuid_generate(s4.uuid);
    cgrad_storage_registry_register(registry, &s4, NULL);
    // No further access to record->entries after record is stopped
    
    // Deregister after stopping - should NOT be removed from tracker (tracker is stopped)
    cgrad_storage_registry_deregister(registry, &s2);
    cgrad_storage_registry_deregister(registry, &s3);
    cgrad_storage_registry_deregister(registry, &s4);
}

static void test_cgrad_storage_registry_tracker_nested(void **state) {
    (void)state;
    cgrad_storage_registry* registry = cgrad_storage_get_global_registry();
    assert_non_null(registry);
    
    // Start first tracker
    cgrad_storage_registry_record* record1 = cgrad_storage_registry_start_recording(registry);
    assert_non_null(record1);
    
    // Register some storages
    cgrad_storage s1 = {0}, s2 = {0};
    uuid_generate(s1.uuid);
    uuid_generate(s2.uuid);
    cgrad_storage_registry_register(registry, &s1, NULL);
    cgrad_storage_registry_register(registry, &s2, NULL);
    
    // Start nested record
    cgrad_storage_registry_record* record2 = cgrad_storage_registry_start_recording(registry);
    assert_non_null(record2);
    
    // Register more storages
    cgrad_storage s3 = {0}, s4 = {0};
    uuid_generate(s3.uuid);
    uuid_generate(s4.uuid);
    cgrad_storage_registry_register(registry, &s3, NULL);
    cgrad_storage_registry_register(registry, &s4, NULL);
    
    // Check record1 has all 4
    assert_int_equal(cgrad_storage_registry_record_count(record1), 4);
    
    // Check record2 has only 2
    assert_int_equal(cgrad_storage_registry_record_count(record2), 2);
    
    // Verify record2 has s3 and s4
    cgrad_storage_registry_node* entry = NULL;
    HASH_FIND(hh, record2->storage_map, s3.uuid, sizeof(uuid_t), entry);
    assert_non_null(entry);
    HASH_FIND(hh, record2->storage_map, s4.uuid, sizeof(uuid_t), entry);
    assert_non_null(entry);
    
    // Deregister s1 while both records are active - should be removed from record1
    cgrad_storage_registry_deregister(registry, &s1);
    assert_int_equal(cgrad_storage_registry_record_count(record1), 3); // Now has s2, s3, s4
    assert_int_equal(cgrad_storage_registry_record_count(record2), 2); // Still has s3, s4
    
    // Stop both records
    assert_int_equal(cgrad_storage_registry_stop_recording(registry, record2), CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_registry_stop_recording(registry, record1), CGRAD_SUCCESS);
    
    // Deregister after stopping - records keep their snapshots
    cgrad_storage_registry_deregister(registry, &s2);
    cgrad_storage_registry_deregister(registry, &s3);
    cgrad_storage_registry_deregister(registry, &s4);
    // No further access to record1->entries or record2->entries after records are stopped
}

int run_cgrad_storage_registry_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_cgrad_storage_register_root_and_find, registry_setup_test, registry_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_register_child_and_bucket_sharing, registry_setup_test, registry_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_register_with_unregistered_parent, registry_setup_test, registry_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_register_count, registry_setup_test, registry_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_register_idempotency, registry_setup_test, registry_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_registry_tracker_basic, registry_setup_test, registry_teardown_test),
        cmocka_unit_test_setup_teardown(test_cgrad_storage_registry_tracker_nested, registry_setup_test, registry_teardown_test),
    };
    return cmocka_run_group_tests_name("cgrad_storage_registry", tests, NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_storage_registry_tests();
}
#endif

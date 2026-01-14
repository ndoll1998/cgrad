#include <cmocka.h>
#include "cgrad_storage_registry.h"
#include "cgrad_storage.h"
#include "cgrad_errors.h"
#include <stdlib.h>

// Forward declare getter - it's internal to storage module
extern cgrad_storage_registry* get_global_registry(void);

static void test_cgrad_storage_register_root_and_find(void **state) {
    (void)state;
    cgrad_storage_registry* registry = get_global_registry();
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
    cgrad_storage_registry_entry_tensor* entry = NULL;
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
    cgrad_storage_cleanup_global_registry();
}

static void test_cgrad_storage_register_child_and_bucket_sharing(void **state) {
    (void)state;
    cgrad_storage_registry* registry = get_global_registry();
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
    cgrad_storage_registry_entry_tensor* entry = NULL;
    HASH_FIND(hh, root_bucket->storage_map, root->uuid, sizeof(uuid_t), entry);
    assert_non_null(entry);
    HASH_FIND(hh, root_bucket->storage_map, child->uuid, sizeof(uuid_t), entry);
    assert_non_null(entry);

    // Deregister child, bucket should not be empty
    rc = cgrad_storage_registry_deregister(registry, child);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_storage_registry_get_bucket_size(registry, root), 1);

    // Deregister root, now bucket should be empty
    rc = cgrad_storage_registry_deregister(registry, root);
    assert_int_equal(rc, CGRAD_SUCCESS);

    free(child);
    free(root);
    cgrad_storage_cleanup_global_registry();
}

static void test_cgrad_storage_register_with_unregistered_parent(void **state) {
    (void)state;
    cgrad_storage_registry* registry = get_global_registry();
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
    cgrad_storage_cleanup_global_registry();
}

static void test_cgrad_storage_register_count(void **state) {
    (void)state;
    cgrad_storage_registry* registry = get_global_registry();
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
    cgrad_storage_cleanup_global_registry();
}

static void test_cgrad_storage_register_idempotency(void **state) {
    (void)state;
    cgrad_storage_registry* registry = get_global_registry();
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
    cgrad_storage_cleanup_global_registry();
}

int run_cgrad_storage_registry_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cgrad_storage_register_root_and_find),
        cmocka_unit_test(test_cgrad_storage_register_child_and_bucket_sharing),
        cmocka_unit_test(test_cgrad_storage_register_with_unregistered_parent),
        cmocka_unit_test(test_cgrad_storage_register_count),
        cmocka_unit_test(test_cgrad_storage_register_idempotency),
    };
    return _cmocka_run_group_tests("cgrad_storage_registry", tests, sizeof(tests)/sizeof(tests[0]), NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_storage_registry_tests();
}
#endif

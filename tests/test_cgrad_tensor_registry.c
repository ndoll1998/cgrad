#include <cmocka.h>
#include "cgrad_tensor_registry.h"
#include "cgrad_tensor.h"
#include "cgrad_errors.h"
#include <stdlib.h>

static void test_register_root_and_find(void **state) {
    (void)state;
    cgrad_tensor* tensor = (cgrad_tensor*)malloc(sizeof(cgrad_tensor));
    assert_non_null(tensor);
    uuid_generate(tensor->uuid);

    int rc = cgrad_tensor_registry_register(tensor, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);

    extern cgrad_tensor_registry global_tensor_registry;
    cgrad_tensor_registry_entry* reg_entry = NULL;
    HASH_FIND(hh, global_tensor_registry.tensor_map, tensor->uuid, sizeof(uuid_t), reg_entry);
    assert_non_null(reg_entry);
    cgrad_tensor_registry_bucket* bucket = reg_entry->bucket;
    assert_non_null(bucket);
    assert_ptr_equal(bucket->root.data, tensor->data);

    // Check tensor is in bucket's tensor_map
    cgrad_tensor_registry_tensor_entry* entry = NULL;
    HASH_FIND(hh, bucket->tensor_map, tensor->uuid, sizeof(uuid_t), entry);
    assert_non_null(entry);
    assert_ptr_equal(entry->tensor, tensor);

    // Deregister and check registry state
    rc = cgrad_tensor_registry_deregister(tensor);
    assert_int_equal(rc, CGRAD_SUCCESS);

    // Deregister again (should return error)
    rc = cgrad_tensor_registry_deregister(tensor);
    assert_int_equal(rc, CGRAD_TENSOR_REGISTRY_PARENT_NOT_REGISTERED);

    free(tensor);
}

static void test_register_child_and_bucket_sharing(void **state) {
    (void)state;
    cgrad_tensor* root = (cgrad_tensor*)malloc(sizeof(cgrad_tensor));
    cgrad_tensor* child = (cgrad_tensor*)malloc(sizeof(cgrad_tensor));
    assert_non_null(root);
    assert_non_null(child);
    uuid_generate(root->uuid);
    uuid_generate(child->uuid);

    int rc = cgrad_tensor_registry_register(root, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);

    rc = cgrad_tensor_registry_register(child, root);
    assert_int_equal(rc, CGRAD_SUCCESS);

    extern cgrad_tensor_registry global_tensor_registry;
    cgrad_tensor_registry_entry* root_entry = NULL;
    cgrad_tensor_registry_entry* child_entry = NULL;
    HASH_FIND(hh, global_tensor_registry.tensor_map, root->uuid, sizeof(uuid_t), root_entry);
    HASH_FIND(hh, global_tensor_registry.tensor_map, child->uuid, sizeof(uuid_t), child_entry);

    assert_non_null(root_entry);
    assert_non_null(child_entry);
    cgrad_tensor_registry_bucket* root_bucket = root_entry->bucket;
    cgrad_tensor_registry_bucket* child_bucket = child_entry->bucket;
    assert_non_null(root_bucket);
    assert_non_null(child_bucket);
    assert_ptr_equal(root_bucket, child_bucket);
    assert_ptr_equal(root_bucket->root.data, root->data);

    // Both tensors should be in the bucket's tensor_map
    cgrad_tensor_registry_tensor_entry* entry = NULL;
    HASH_FIND(hh, root_bucket->tensor_map, root->uuid, sizeof(uuid_t), entry);
    assert_non_null(entry);
    HASH_FIND(hh, root_bucket->tensor_map, child->uuid, sizeof(uuid_t), entry);
    assert_non_null(entry);

    // Deregister child, bucket should not be empty
    rc = cgrad_tensor_registry_deregister(child);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_tensor_registry_get_bucket_size(root), 1);

    // Deregister root, now bucket should be empty
    rc = cgrad_tensor_registry_deregister(root);
    assert_int_equal(rc, CGRAD_SUCCESS);

    free(child);
    free(root);
}

static void test_register_with_unregistered_parent(void **state) {
    (void)state;
    cgrad_tensor* parent = (cgrad_tensor*)malloc(sizeof(cgrad_tensor));
    cgrad_tensor* child = (cgrad_tensor*)malloc(sizeof(cgrad_tensor));
    assert_non_null(parent);
    assert_non_null(child);
    uuid_generate(parent->uuid);
    uuid_generate(child->uuid);

    // Do not register parent
    int rc = cgrad_tensor_registry_register(child, parent);
    assert_int_equal(rc, CGRAD_TENSOR_REGISTRY_PARENT_NOT_REGISTERED);

    free(child);
    free(parent);
}

static void test_registry_count(void **state) {
    (void)state;
    // Should be empty at start
    assert_int_equal(cgrad_tensor_registry_count(), 0);

    cgrad_tensor* t1 = (cgrad_tensor*)malloc(sizeof(cgrad_tensor));
    cgrad_tensor* t2 = (cgrad_tensor*)malloc(sizeof(cgrad_tensor));
    assert_non_null(t1);
    assert_non_null(t2);
    uuid_generate(t1->uuid);
    uuid_generate(t2->uuid);

    // Register t1
    int rc = cgrad_tensor_registry_register(t1, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_tensor_registry_count(), 1);

    // Register t2
    rc = cgrad_tensor_registry_register(t2, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_tensor_registry_count(), 2);

    // Deregister t1
    rc = cgrad_tensor_registry_deregister(t1);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_tensor_registry_count(), 1);

    // Deregister t2
    rc = cgrad_tensor_registry_deregister(t2);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_tensor_registry_count(), 0);

    free(t1);
    free(t2);
}

static void test_idempotency(void **state) {
    (void)state;
    cgrad_tensor* tensor = (cgrad_tensor*)malloc(sizeof(cgrad_tensor));
    assert_non_null(tensor);
    uuid_generate(tensor->uuid);

    // Register twice as root
    int rc = cgrad_tensor_registry_register(tensor, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);
    rc = cgrad_tensor_registry_register(tensor, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);

    extern cgrad_tensor_registry global_tensor_registry;
    cgrad_tensor_registry_entry* reg_entry = NULL;
    HASH_FIND(hh, global_tensor_registry.tensor_map, tensor->uuid, sizeof(uuid_t), reg_entry);
    assert_non_null(reg_entry);

    // Deregister twice
    rc = cgrad_tensor_registry_deregister(tensor);
    assert_int_equal(rc, CGRAD_SUCCESS);

    rc = cgrad_tensor_registry_deregister(tensor);
    assert_int_equal(rc, CGRAD_TENSOR_REGISTRY_PARENT_NOT_REGISTERED);

    free(tensor);
}

int run_cgrad_tensor_registry_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_register_root_and_find),
        cmocka_unit_test(test_register_child_and_bucket_sharing),
        cmocka_unit_test(test_register_with_unregistered_parent),
        cmocka_unit_test(test_registry_count),
        cmocka_unit_test(test_idempotency),
    };
    return _cmocka_run_group_tests("cgrad_tensor_registry", tests, sizeof(tests)/sizeof(tests[0]), NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_tensor_registry_tests();
}
#endif

#include <cmocka.h>
#include "cgrad_tensor_registry.h"
#include "cgrad_tensor.h"
#include "cgrad_errors.h"
#include <stdlib.h>

static void test_register_root_and_find(void **state) {
    (void)state;
    cgrad_tensor* tensor = (cgrad_tensor*)malloc(sizeof(cgrad_tensor));
    assert_non_null(tensor);

    int rc = cgrad_tensor_registry_register(tensor, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);

    extern cgrad_tensor_registry global_tensor_registry;
    cgrad_tensor_registry_entry* reg_entry = NULL;
    HASH_FIND_PTR(global_tensor_registry.tensor_map, &tensor, reg_entry);
    assert_non_null(reg_entry);
    cgrad_tensor_registry_bucket* bucket = reg_entry->bucket;
    assert_non_null(bucket);
    assert_ptr_equal(bucket->root, tensor);

    // Check tensor is in bucket's tensor_map
    cgrad_tensor_registry_tensor_entry* entry = NULL;
    HASH_FIND_PTR(bucket->tensor_map, &tensor, entry);
    assert_non_null(entry);
    assert_ptr_equal(entry->tensor, tensor);

    // Deregister and check root output
    cgrad_tensor* out_root = NULL;
    rc = cgrad_tensor_registry_deregister(tensor, &out_root);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_ptr_equal(out_root, tensor);

    // Deregister again (should return error)
    rc = cgrad_tensor_registry_deregister(tensor, &out_root);
    assert_int_equal(rc, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED);

    free(tensor);
}

static void test_register_child_and_bucket_sharing(void **state) {
    (void)state;
    cgrad_tensor* root = (cgrad_tensor*)malloc(sizeof(cgrad_tensor));
    cgrad_tensor* child = (cgrad_tensor*)malloc(sizeof(cgrad_tensor));
    assert_non_null(root);
    assert_non_null(child);

    int rc = cgrad_tensor_registry_register(root, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);

    rc = cgrad_tensor_registry_register(child, root);
    assert_int_equal(rc, CGRAD_SUCCESS);

    extern cgrad_tensor_registry global_tensor_registry;
    cgrad_tensor_registry_entry* root_entry = NULL;
    cgrad_tensor_registry_entry* child_entry = NULL;
    HASH_FIND_PTR(global_tensor_registry.tensor_map, &root, root_entry);
    HASH_FIND_PTR(global_tensor_registry.tensor_map, &child, child_entry);

    assert_non_null(root_entry);
    assert_non_null(child_entry);
    cgrad_tensor_registry_bucket* root_bucket = root_entry->bucket;
    cgrad_tensor_registry_bucket* child_bucket = child_entry->bucket;
    assert_non_null(root_bucket);
    assert_non_null(child_bucket);
    assert_ptr_equal(root_bucket, child_bucket);
    assert_ptr_equal(root_bucket->root, root);

    // Both tensors should be in the bucket's tensor_map
    cgrad_tensor_registry_tensor_entry* entry = NULL;
    HASH_FIND_PTR(root_bucket->tensor_map, &root, entry);
    assert_non_null(entry);
    HASH_FIND_PTR(root_bucket->tensor_map, &child, entry);
    assert_non_null(entry);

    // Deregister child, bucket should not be empty
    cgrad_tensor* out_root = (cgrad_tensor*)0xDEADBEEF;
    rc = cgrad_tensor_registry_deregister(child, &out_root);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_ptr_equal(out_root, NULL);

    // Deregister root, now bucket should be empty and out_root should be root
    rc = cgrad_tensor_registry_deregister(root, &out_root);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_ptr_equal(out_root, root);

    free(child);
    free(root);
}

static void test_register_with_unregistered_parent(void **state) {
    (void)state;
    cgrad_tensor* parent = (cgrad_tensor*)malloc(sizeof(cgrad_tensor));
    cgrad_tensor* child = (cgrad_tensor*)malloc(sizeof(cgrad_tensor));
    assert_non_null(parent);
    assert_non_null(child);

    // Do not register parent
    int rc = cgrad_tensor_registry_register(child, parent);
    assert_int_equal(rc, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED);

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

    // Register t1
    int rc = cgrad_tensor_registry_register(t1, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_tensor_registry_count(), 1);

    // Register t2
    rc = cgrad_tensor_registry_register(t2, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_tensor_registry_count(), 2);

    // Deregister t1
    cgrad_tensor* out_root = NULL;
    rc = cgrad_tensor_registry_deregister(t1, &out_root);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_tensor_registry_count(), 1);

    // Deregister t2
    rc = cgrad_tensor_registry_deregister(t2, &out_root);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_int_equal(cgrad_tensor_registry_count(), 0);

    free(t1);
    free(t2);
}

static void test_idempotency(void **state) {
    (void)state;
    cgrad_tensor* tensor = (cgrad_tensor*)malloc(sizeof(cgrad_tensor));
    assert_non_null(tensor);

    // Register twice as root
    int rc = cgrad_tensor_registry_register(tensor, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);
    rc = cgrad_tensor_registry_register(tensor, NULL);
    assert_int_equal(rc, CGRAD_SUCCESS);

    extern cgrad_tensor_registry global_tensor_registry;
    cgrad_tensor_registry_entry* reg_entry = NULL;
    HASH_FIND_PTR(global_tensor_registry.tensor_map, &tensor, reg_entry);
    assert_non_null(reg_entry);

    // Deregister twice
    cgrad_tensor* out_root = NULL;
    rc = cgrad_tensor_registry_deregister(tensor, &out_root);
    assert_int_equal(rc, CGRAD_SUCCESS);
    assert_ptr_equal(out_root, tensor);

    rc = cgrad_tensor_registry_deregister(tensor, &out_root);
    assert_int_equal(rc, CGRAD_TENSOR_ERR_PARENT_NOT_REGISTERED);

    free(tensor);
}

int run_cgrad_tensor_registry_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_register_root_and_find),
        cmocka_unit_test(test_register_child_and_bucket_sharing),
        cmocka_unit_test(test_register_with_unregistered_parent),
        cmocka_unit_test(test_idempotency),
        cmocka_unit_test(test_registry_count),
    };
    return _cmocka_run_group_tests("cgrad_tensor_registry", tests, sizeof(tests)/sizeof(tests[0]), NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_tensor_registry_tests();
}
#endif

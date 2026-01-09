#include <cmocka.h>
#include "cgrad_tensor.h"
#include "backends/cgrad_tensor_f32_cpu.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define MAX_TENSOR_DIM 4

static void test_cgrad_tensor_init_and_free(void **state) {
    (void)state;
    cgrad_tensor t;
    uint32_t shape[MAX_TENSOR_DIM] = {2, 3, 4, 5};
    assert_int_equal(cgrad_tensor_init(&t, shape, CGRAD_BACKEND_F32_CPU), 0);
    assert_non_null(t.handle);
    cgrad_tensor_free(&t);
    assert_null(t.handle);
}

int run_cgrad_tensor_tests(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cgrad_tensor_init_and_free),
    };
    return _cmocka_run_group_tests("cgrad_tensor", tests, sizeof(tests)/sizeof(tests[0]), NULL, NULL);
}

#ifndef TEST_ALL_MAIN
int main(void) {
    return run_cgrad_tensor_tests();
}
#endif

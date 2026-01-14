#define TEST_ALL_MAIN
#include "test_cgrad_storage_layout.c"
#include "test_cgrad_storage.c"
#include "storage_backends/test_cgrad_storage_f32_cpu.c"
#include "test_cgrad_storage_registry.c"
#include "test_cgrad_compute_graph.c"
#include "test_cgrad_tensor.c"
#include "ops/test_cgrad_op_add.c"
#include "ops/test_cgrad_op_sub.c"
#include "ops/test_cgrad_op_gemm.c"
#include "ops/test_cgrad_op_transpose.c"
#include "ops/test_cgrad_op_reshape.c"
#include "ops/test_cgrad_op_reduce_sum.c"

int main(void) {
    int failed = 0;
    failed |= run_cgrad_storage_layout_tests();
    failed |= run_cgrad_storage_tests();
    failed |= run_cgrad_storage_f32_cpu_tests();
    failed |= run_cgrad_storage_registry_tests();
    failed |= test_cgrad_compute_graph_main();
    failed |= test_cgrad_tensor_main();
    failed |= test_cgrad_op_add_main();
    failed |= test_cgrad_op_sub_main();
    failed |= test_cgrad_op_gemm_main();
    failed |= test_cgrad_op_transpose_main();
    failed |= test_cgrad_op_reshape_main();
    failed |= test_cgrad_op_reduce_sum_main();
    return failed;
}

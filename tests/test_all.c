#define TEST_ALL_MAIN
#include "storage/test_cgrad_storage_layout.c"
#include "storage/test_cgrad_storage.c"
#include "backends/cpu/test_cgrad_backend_cpu_f32.c"
#include "storage/test_cgrad_storage_registry.c"
#include "autograd/test_cgrad_compute_graph.c"
#include "autograd/test_cgrad_tensor.c"
#include "autograd/ops/test_cgrad_op_axpy.c"
#include "autograd/ops/test_cgrad_op_gemm.c"
#include "autograd/ops/test_cgrad_op_transpose.c"
#include "autograd/ops/test_cgrad_op_reshape.c"
#include "autograd/ops/test_cgrad_op_reduce_sum.c"

int main(void) {
    int failed = 0;
    failed |= run_cgrad_storage_layout_tests();
    failed |= run_cgrad_storage_tests();
    failed |= run_cgrad_backend_cpu_f32_tests();
    failed |= run_cgrad_storage_registry_tests();
    failed |= run_cgrad_compute_graph_tests();
    failed |= run_cgrad_tensor_tests();
    failed |= run_cgrad_op_axpy_tests();
    failed |= run_cgrad_op_gemm_tests();
    failed |= run_cgrad_op_transpose_tests();
    failed |= run_cgrad_op_reshape_tests();
    failed |= run_cgrad_op_reduce_sum_tests();
    return failed;
}

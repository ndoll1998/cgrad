#define TEST_ALL_MAIN
#include "test_cgrad_layout.c"
#include "test_cgrad_tensor.c"
#include "backends/test_cgrad_tensor_f32_cpu.c"
#include "test_cgrad_tensor_registry.c"

int main(void) {
    int failed = 0;
    failed |= run_cgrad_layout_tests();
    failed |= run_cgrad_tensor_tests();
    failed |= run_cgrad_tensor_f32_cpu_tests();
    failed |= run_cgrad_tensor_registry_tests();
    return failed;
}

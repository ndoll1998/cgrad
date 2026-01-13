#define TEST_ALL_MAIN
#include "test_cgrad_storage_layout.c"
#include "test_cgrad_storage.c"
#include "storage_backends/test_cgrad_storage_f32_cpu.c"
#include "test_cgrad_storage_registry.c"

int main(void) {
    int failed = 0;
    failed |= run_cgrad_storage_layout_tests();
    failed |= run_cgrad_storage_tests();
    failed |= run_cgrad_storage_f32_cpu_tests();
    failed |= run_cgrad_storage_registry_tests();
    return failed;
}

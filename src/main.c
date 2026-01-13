#include "cgrad_storage.h"
#include "cgrad_storage_registry.h"
#include "storage_backends/cgrad_storage_f32_cpu.h"
#include <stdio.h>
#include <stdint.h>

/**
 * @brief Example program demonstrating tensor initialization, randomization, printing,
 *        transposition, GEMM, addition, and memory management using the cgrad library.
 */
int main() {
  cgrad_storage t1 = {0};
  cgrad_storage t2 = {0};
  cgrad_storage out = {0};
  cgrad_storage r = {0};
  
  // initialize tensors (CPU backend)
  cgrad_storage_init(&t1, (uint32_t[]){1, 2, 3, 3}, 4, CGRAD_STORAGE_BACKEND_F32_CPU);
  cgrad_storage_init(&t2, (uint32_t[]){1, 1, 3, 3}, 4, CGRAD_STORAGE_BACKEND_F32_CPU);

  // fill with random numbers
  cgrad_storage_fill_rand(&t1);
  cgrad_storage_fill_rand(&t2);

  // print, transpose, print
  cgrad_storage_transpose(&t1, (uint32_t[]){0,1,3,2}, 4);

  // GEMM
  int e = cgrad_storage_gemm(&t1, &t2, &out);
  printf("GEMM error code: %d\n", e);
  cgrad_storage_print(&out);

  // Add output to itself
  //e = cgrad_storage_add(&out, &out, &r);
  //printf("Add error code: %d\n", e);
  //cgrad_storage_print(&r);

  e = cgrad_storage_sum(
    &out, (const uint8_t[]){1,1,1}, 3, &r
  );
  printf("Sum error code: %d\n", e);
  cgrad_storage_print(&r);

  size_t n = cgrad_storage_registry_count();
  printf("Number of tensors in registry: %zu\n", n);

  // free
  cgrad_storage_free(&t1);
  cgrad_storage_free(&t2);
  cgrad_storage_free(&out);
  cgrad_storage_free(&r);

  n = cgrad_storage_registry_count();
  printf("Number of tensors in registry: %zu\n", n);

  return 0;
}

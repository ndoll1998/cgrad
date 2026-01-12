#include "cgrad_tensor.h"
#include "cgrad_tensor_registry.h"
#include "backends/cgrad_tensor_f32_cpu.h"
#include <stdio.h>
#include <stdint.h>

/**
 * @brief Example program demonstrating tensor initialization, randomization, printing,
 *        transposition, GEMM, addition, and memory management using the cgrad library.
 */
int main() {
  cgrad_tensor t1 = {0};
  cgrad_tensor t2 = {0};
  cgrad_tensor out = {0};
  cgrad_tensor r = {0};
  
  // initialize tensors (CPU backend)
  cgrad_tensor_init(&t1, (uint32_t[]){1, 2, 3, 3}, 4, CGRAD_BACKEND_F32_CPU);
  cgrad_tensor_init(&t2, (uint32_t[]){1, 1, 3, 3}, 4, CGRAD_BACKEND_F32_CPU);

  // fill with random numbers
  cgrad_tensor_fill_rand(&t1);
  cgrad_tensor_fill_rand(&t2);

  // print, transpose, print
  cgrad_tensor_transpose(&t1, (uint32_t[]){0,1,3,2}, 4);

  // GEMM
  int e = cgrad_tensor_gemm(&t1, &t2, &out);
  printf("GEMM error code: %d\n", e);
  cgrad_tensor_print(&out);

  // Add output to itself
  //e = cgrad_tensor_add(&out, &out, &r);
  //printf("Add error code: %d\n", e);
  //cgrad_tensor_print(&r);

  e = cgrad_tensor_sum(
    &out, (const uint8_t[]){1,1,1}, 3, &r
  );
  printf("Sum error code: %d\n", e);
  cgrad_tensor_print(&r);

  size_t n = cgrad_tensor_registry_count();
  printf("Number of tensors in registry: %zu\n", n);

  // free
  cgrad_tensor_free(&t1);
  cgrad_tensor_free(&t2);
  cgrad_tensor_free(&out);
  cgrad_tensor_free(&r);

  n = cgrad_tensor_registry_count();
  printf("Number of tensors in registry: %zu\n", n);

  return 0;
}

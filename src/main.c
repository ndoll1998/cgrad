#include "cgrad_tensor.h"
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
  e = cgrad_tensor_add(&out, &out, &r);
  printf("Add error code: %d\n", e);
  cgrad_tensor_print(&r);

  // free
  cgrad_tensor_free(&t1);
  cgrad_tensor_free(&t2);
  cgrad_tensor_free(&out);

  return 0;
}

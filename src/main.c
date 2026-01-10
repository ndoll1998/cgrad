#include "cgrad_tensor.h"
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
  uint32_t shapeA[] = {1, 2, 2, 3};
  uint32_t shapeB[] = {1, 1, 2, 3};

  // initialize tensors (CPU backend)
  cgrad_tensor_init(&t1, shapeA, CGRAD_BACKEND_F32_CPU);
  cgrad_tensor_init(&t2, shapeB, CGRAD_BACKEND_F32_CPU);

  // fill with random numbers
  cgrad_tensor_fill_rand(&t1);
  cgrad_tensor_fill_rand(&t2);

  // print, transpose, print
  cgrad_tensor_print(&t1);
  cgrad_tensor_transpose(&t1, (uint32_t[]){0,1,3,2});
  cgrad_tensor_print(&t1);

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
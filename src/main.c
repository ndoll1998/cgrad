#include "cgrad_tensor.h"
#include <stdio.h>
#include <stdint.h>

int main() {
  cgrad_tensor t1, t2, out;
  uint32_t shapeA[] = {1, 1, 2, 3};
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
  cgrad_tensor_init(&out, (uint32_t[]){1,2,2,2}, CGRAD_BACKEND_F32_CPU); // shape is a placeholder, will be overwritten by gemm
  int e = cgrad_tensor_gemm(&t1, &t2, &out);
  printf("GEMM error code: %d\n", e);
  cgrad_tensor_print(&out);

  // free
  cgrad_tensor_free(&t1);
  cgrad_tensor_free(&t2);
  cgrad_tensor_free(&out);

  return 0;
}

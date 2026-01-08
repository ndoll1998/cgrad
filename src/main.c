#include "cgrad_tensor.h"
#include <stdio.h>
#include <stdint.h>

int main() {
  cgrad_tensor_f32 t1, t2, out;
  uint32_t shapeA[] = {1, 1, 2, 3};
  uint32_t shapeB[] = {1, 1, 3, 2};
  // initialize tensors
  cgrad_tensor_f32_init(&t1, shapeA);
  cgrad_tensor_f32_init(&t2, shapeB);
  // fill with random numbers
  cgrad_tensor_f32_fill_rand(&t1);
  cgrad_tensor_f32_fill_rand(&t2);
  // 
  cgrad_tensor_f32_print(&t1);
  cgrad_tensor_f32_print(&t2);

  int e = cgrad_tensor_f32_gemm(&t1, &t2, &out);
  printf("%i", e);

  // free
  cgrad_tensor_f32_free(&t1);
  cgrad_tensor_f32_free(&t2);
  cgrad_tensor_f32_free(&out);
  return 0;
}

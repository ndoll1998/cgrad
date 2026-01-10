#include "cgrad_backend.h"
#include "backends/cgrad_tensor_f32_cpu.h"

cgrad_backend* cgrad_get_backend(cgrad_backend_type type) {
    switch (type) {
        case CGRAD_BACKEND_F32_CPU:
            return &cgrad_backend_f32_cpu;
        // Add more backends here
        default:
            return NULL;
    }
}

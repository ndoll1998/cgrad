#include "storage/cgrad_storage_backend.h"
#include "storage/backends/cgrad_storage_f32_cpu.h"

cgrad_storage_backend* cgrad_get_backend(cgrad_storage_backend_type type) {
    switch (type) {
        case CGRAD_STORAGE_BACKEND_F32_CPU:
            return &cgrad_storage_backend_f32_cpu;
        // Add more backends here
        default:
            return NULL;
    }
}

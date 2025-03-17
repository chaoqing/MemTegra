#include "MemTegra/Utils.h"

#include <cstdlib>
#include <stdexcept>

void *alignedMalloc(size_t size) {
    if (size == 0) {
        return nullptr;
    }

    void *ptr = nullptr;
    if (posix_memalign(&ptr, 64, size) != 0) {
        throw std::runtime_error("Memory allocation failed");
    }
    return ptr;
}

void alignedFree(void *ptr) { free(ptr); }
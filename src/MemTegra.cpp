#include "MemTegra/MemTegra.h"
#include <cstdlib>
#include <stdexcept>

void* MemTegra::mallocAligned(size_t size) {
    if (size == 0) {
        throw std::invalid_argument("Size must be greater than zero.");
    }
    
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, size) != 0) {
        throw std::runtime_error("Memory allocation failed.");
    }
    return ptr;
}

void MemTegra::freeAligned(void* ptr) {
    if (ptr != nullptr) {
        free(ptr);
    }
}
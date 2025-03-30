#include <iostream>

#include "MemTegra/MemTegra.h"

int main() {
    MT::RawAllocator<MT::MemoryTag::host_aligned_64> memTegra;

    // Allocate aligned memory
    void *ptr = memTegra.malloc(128);
    if (ptr) {
        std::cout << "Memory allocated at address: " << ptr << std::endl;
    } else {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }

    // Free the allocated memory
    memTegra.free(ptr);
    std::cout << "Memory freed successfully." << std::endl;

    return 0;
}

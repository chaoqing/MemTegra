#include "MemTegra/MemTegra.h"
#include <iostream>

int main() {
    MemTegra memTegra;

    // Allocate aligned memory
    void* ptr = memTegra.mallocAligned(128);
    if (ptr) {
        std::cout << "Memory allocated at address: " << ptr << std::endl;
    } else {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }

    // Free the allocated memory
    memTegra.freeAligned(ptr);
    std::cout << "Memory freed successfully." << std::endl;

    return 0;
}

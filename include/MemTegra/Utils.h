// Utils.h - Declaration of utility functions for memory management and other helpers

#ifndef MEMTEGRA_UTILS_H
#define MEMTEGRA_UTILS_H

#include <cstddef>

// Function to check if a pointer is 64-byte aligned
bool isAligned64(void* ptr);

// Function to align a size to the next 64-byte boundary
std::size_t alignTo64(std::size_t size);

#endif // MEMTEGRA_UTILS_H
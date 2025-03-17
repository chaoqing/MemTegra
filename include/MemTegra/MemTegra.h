// MemTegra.h
#ifndef MEMTEGRA_H
#define MEMTEGRA_H

#include <cstddef>
#include <cstdlib>
#include <new>

class MemTegra {
public:
    static void* mallocAligned(std::size_t size);
    static void  freeAligned(void* ptr);
};

#endif  // MEMTEGRA_H
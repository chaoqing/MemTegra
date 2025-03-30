#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "MemTegra/MemTegra.h"
using namespace MT;

class MemTegraTest : public ::testing::Test {
protected:
    RawAllocator<MemoryTag::host_aligned_64> memTegra;
};

TEST_F(MemTegraTest, AllocateAlignedMemory) {
    void *ptr = memTegra.malloc(64);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);
    memTegra.free(ptr);
}

TEST_F(MemTegraTest, AllocateMultipleAlignedMemory) {
    void *ptr1 = memTegra.malloc(128);
    void *ptr2 = memTegra.malloc(256);

    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr1) % 64, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr2) % 64, 0);

    memTegra.free(ptr1);
    memTegra.free(ptr2);
}

TEST_F(MemTegraTest, FreeNullPointer) { EXPECT_NO_THROW(memTegra.free(nullptr)); }


TEST_F(MemTegraTest, MemTegraAllocator) {
    std::vector<int, MemTegraAllocator<int, MemoryTag::host>> vec;

    vec.push_back(10);
    vec.push_back(20);
    vec.push_back(30);

    for (const auto &val : vec) {
        std::cout << val << std::endl;
    }
}
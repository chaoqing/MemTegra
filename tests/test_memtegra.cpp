#include <gtest/gtest.h>
#include "MemTegra/MemTegra.h"

class MemTegraTest : public ::testing::Test {
protected:
    MemTegra memTegra;
};

TEST_F(MemTegraTest, AllocateAlignedMemory) {
    void* ptr = memTegra.mallocAligned(64);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);
    memTegra.freeAligned(ptr);
}

TEST_F(MemTegraTest, AllocateMultipleAlignedMemory) {
    void* ptr1 = memTegra.mallocAligned(128);
    void* ptr2 = memTegra.mallocAligned(256);
    
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr1) % 64, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr2) % 64, 0);
    
    memTegra.freeAligned(ptr1);
    memTegra.freeAligned(ptr2);
}

TEST_F(MemTegraTest, FreeNullPointer) {
    EXPECT_NO_THROW(memTegra.freeAligned(nullptr));
}
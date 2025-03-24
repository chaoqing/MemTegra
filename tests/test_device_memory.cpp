#if ENABLE_CUDA
#    include <gtest/gtest.h>

#    include <iostream>
#    include <vector>

#    include "MemTegra/device_memory.h"
#    include "MemTegra/strong_pointer.hpp"
using namespace MT;

class DeviceMemTest : public ::testing::Test {
protected:
    RawAllocator<MemoryTag::ENUM_DEVICE> memTegra;
};

TEST_F(DeviceMemTest, AllocateDeviceMemory) {
    void *ptr = memTegra.malloc(64);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);
    memTegra.free(ptr);
}

TEST_F(DeviceMemTest, AllocateMultipleDeviceMemory) {
    void *ptr1 = memTegra.malloc(128);
    void *ptr2 = memTegra.malloc(256);

    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr1) % 64, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr2) % 64, 0);

    memTegra.free(ptr1);
    memTegra.free(ptr2);
}

TEST_F(DeviceMemTest, FreeNullPointer) { EXPECT_NO_THROW(memTegra.free(nullptr)); }

TEST_F(DeviceMemTest, TypeSafety) {
    int   *p = static_cast<int *>(memTegra.malloc(128));
    int_dp a(p);

    // Valid operations
    auto a2   = a + 1;
    auto diff = a2 - a;
    EXPECT_EQ(diff, 1);

    // Device memory do not support reference
    //*a = 0;
}
#endif

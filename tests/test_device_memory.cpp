#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "MemTegra/device_memory.h"
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
    memTegra.free(p);
}

TEST_F(DeviceMemTest, MemoryOps) {
    constexpr size_t N     = 100;
    constexpr size_t bytes = N * sizeof(int);
    int_hp           host_p{static_cast<int *>(RawAllocator<MemoryTag::ENUM_HOST>::malloc(bytes))};
    void_dp          device_p{static_cast<int *>(memTegra.malloc(bytes))};

    MT::memset{}(host_p, 0, bytes);
    std::copy(host_p, host_p + N, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // Set first half of device_p to all 1
    MT::memset{}(device_p, 1, bytes / 2);

    // Copy first half of device_p into later half of host_p
    MT::memcpy{}(host_p + bytes / 2, device_p, bytes / 2);

    // Print all values in host_p
    std::copy(host_p, host_p + N, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // Verify values in host_p
    for (size_t i = N / 2; i < N; ++i) {
        EXPECT_EQ(host_p[i], 0x01010101);  // Each byte is set to 1
    }

    // Generate random numbers in first half of host_p
    std::generate(host_p, host_p + N / 2, []() { return rand(); });

    // Copy first half of host_p into later half of device_p
    MT::memcpy{}(static_cast<int_dp>(device_p) + bytes / 2, host_p, bytes / 2);

    memTegra.free(device_p.get());
    RawAllocator<MemoryTag::ENUM_HOST>::free(host_p.get());
}

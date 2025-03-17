#include <gtest/gtest.h>

#include "MemTegra/strong_pointer.hpp"

using namespace MT;

// Test basic functionality
TEST(StrongPointerTest, BasicFunctionality) {
    int    hostArray[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int_hp a(hostArray);

    // Test dereference
    EXPECT_EQ(*a, 0);

    // Test pointer arithmetic
    auto a2 = a + 1;
    EXPECT_EQ(*a2, 1);

    // Test indexing
    EXPECT_EQ(a[2], 2);
}

// Test arithmetic operations
TEST(StrongPointerTest, ArithmeticOperations) {
    int    hostArray[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int_hp a(hostArray);

    // Test addition
    auto a2 = a + 3;
    EXPECT_EQ(*a2, 3);

    // Test subtraction
    auto diff = a2 - a;
    EXPECT_EQ(diff, 3);

    // Test compound addition
    a += 2;
    EXPECT_EQ(*a, 2);

    // Test compound subtraction
    a -= 1;
    EXPECT_EQ(*a, 1);
}

// Test comparison operations
TEST(StrongPointerTest, ComparisonOperations) {
    int    hostArray[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int_hp a(hostArray);
    int_hp b(hostArray + 1);

    // Test equality
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);

    // Test less than
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);

    // Test greater than
    EXPECT_TRUE(b > a);
    EXPECT_FALSE(a > b);

    // Test less than or equal
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a <= a);

    // Test greater than or equal
    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(b >= b);
}

// Test type safety
TEST(StrongPointerTest, TypeSafety) {
    int hostArray[10];
    int deviceArray[10];

    int_hp a(hostArray);
    int_dp b(deviceArray);

    // Valid operations
    auto a2   = a + 1;
    auto diff = a2 - a;
    EXPECT_EQ(diff, 1);

    EXPECT_EQ(a + (b - b), a);
    // Invalid operations (should not compile, so we can't test them directly)
    // Uncommenting the following lines should cause compile-time errors:
    // auto invalid = a - b;
}
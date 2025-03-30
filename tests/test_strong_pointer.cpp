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
TEST(StrongPointerTest, VoidPointer) {
    int_hp a;
    void  *p = a.get();

    void_hp b(p);
    void_hp c = a;
    a         = static_cast<int_hp>(c);
    EXPECT_EQ(b, nullptr);
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
    using int_hp_64  = strong_pointer<int, MemoryTag::host_aligned_64>;
    using void_hp_64 = strong_pointer<void, MemoryTag::host_aligned_64>;
    int_hp_64 b(deviceArray);

    // Valid operations
    auto a2   = a + 1;
    auto diff = a2 - a;
    EXPECT_EQ(diff, 1);

    EXPECT_EQ(a + (b - b), a);
    // Invalid operations (should not compile, so we can't test them directly)
    // Uncommenting the following lines should cause compile-time errors:
    // auto invalid = a - b;

    void_hp_64 c = b;
    // void pointer do not support reference
    //*c = 0;

    void_hp d = b;
    // convert from (int*, host_aligned_64) into (void*, host) allowed
    // c = d;
    // b = d;
}


struct TagBase {};
struct TagA : TagBase {};
struct TagB : TagBase {};
struct Base {
    int x;
};
struct Derived : Base {
    int y;
};

// Test new
TEST(StrongPointerTest, MoreTestCases) {
    // Test non-void case (T = int)
    int                       x = 42;
    strong_pointer<int, TagA> pa(&x);
    assert(*pa == 42);  // Dereference works
    pa = pa + 1;
    assert(pa.get() == &x + 1);  // Arithmetic works

    // Test void case
    strong_pointer<void, TagA> pv(static_cast<void *>(&x));
    assert(pv.get() == static_cast<void *>(&x));
    pv = pa - 1;  // any typed pointer can be assigned to strong void pointer with the same tag
    pv = pv + 1;
    assert(reinterpret_cast<char *>(pv.get()) == reinterpret_cast<char *>(&x) + 1);
    // Note: *pv would not compile, which is correct behavior

    // Test with standard algorithm (non-void)
    int                       src[3]  = {1, 2, 3};
    int                       dest[3] = {0, 0, 0};
    strong_pointer<int, TagA> src_begin(src);
    strong_pointer<int, TagA> src_end(src + 3);
    strong_pointer<int, TagA> dest_begin(dest);
    std::copy(src_begin, src_end, dest_begin);
    assert(dest[0] == 1 && dest[1] == 2 && dest[2] == 3);

    // Test tag safety (should not compile if uncommented)
    strong_pointer<int, TagB> pb(&x);
    // pa = pb;  // Error: different tags
    // assert(pa == pb);  // Error: different tags

    // Test inheritance
    Derived                       d;
    Base                         *_x = &d;
    Derived                      *_y = static_cast<Derived *>(_x);
    strong_pointer<Derived, TagA> pd(&d);
    strong_pointer<Base, TagA>    pbb = pd;  // OK: same tag, convertible types
    assert(pbb->x == d.x);
    pd = static_cast<strong_pointer<Derived, TagA>>(pbb);

    // Test null pointer operations
    strong_pointer<int, TagA> null_ptr;
    assert(!null_ptr);            // operator bool
    assert(null_ptr == nullptr);  // Comparison with nullptr_t
    // assert(null_ptr != &x);  // Comparison with non-null
    null_ptr = nullptr;  // Assignment from nullptr_t
    assert(null_ptr.get() == nullptr);

    strong_pointer<int, TagA> non_null(&x);
    assert(non_null != nullptr);  // Comparison with nullptr_t
    // assert(non_null == &x);
    non_null = nullptr;  // Assignment from nullptr_t
    assert(non_null == nullptr);
}

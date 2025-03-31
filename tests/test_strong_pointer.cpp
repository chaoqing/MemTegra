#include <gtest/gtest.h>

#include <cstddef>

#include "MemTegra/strong_pointer.hpp"

using namespace MT;

class StrongPointerTest : public ::testing::Test {
protected:
    struct TagBase {};
    struct TagA : TagBase {};
    struct TagB : TagBase {};
    struct Base {
        int x;
    };
    struct Derived : Base {
        int y;
    };
};

// Test basic functionality
TEST_F(StrongPointerTest, BasicFunctionality) {
    int    hostArray[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int_hp a(hostArray);

    // Test dereference
    EXPECT_EQ(*a, 0);

    // Test pointer arithmetic
    auto a2 = a + 1;
    EXPECT_EQ(*a2, 1);

    // Test indexing
    EXPECT_EQ(a[2], 2);

    // Test I/O
    std::stringstream ss;
    ss << a;
    std::cout << "pointer of a:" << ss.str() << ", a2: " << a2 << std::endl;
    ss >> std::hex >> a2;
    EXPECT_EQ(a, a2);
}

// Test arithmetic operations
TEST_F(StrongPointerTest, VoidPointer) {
    int_hp a;
    void*  p = a.get();

    void_hp b(p);
    void_hp c = a;
    a         = static_cast<int_hp>(c);
    EXPECT_EQ(b, nullptr);
    EXPECT_EQ(a, c.cast_static<int_hp>());
}

// Test arithmetic operations
TEST_F(StrongPointerTest, ArithmeticOperations) {
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
TEST_F(StrongPointerTest, ComparisonOperations) {
    int    hostArray[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int_hp a(hostArray);
    int_hp b(hostArray + 1);
    void*  c = hostArray;

    // Test equality with raw pointer
    EXPECT_TRUE(a == hostArray);
    EXPECT_TRUE(hostArray == a);
    EXPECT_TRUE(c == hostArray);
    EXPECT_TRUE(hostArray == c);
    EXPECT_TRUE(c == a);
    EXPECT_TRUE(a == c);

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
TEST_F(StrongPointerTest, TypeSafety) {
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

// Test new
TEST_F(StrongPointerTest, MoreTestCases) {
    // Test non-void case (T = int)
    int                       x = 42;
    strong_pointer<int, TagA> pa(&x);
    assert(*pa == 42);  // Dereference works
    pa = pa + 1;
    assert(pa.get() == &x + 1);  // Arithmetic works

    // Test void case
    strong_pointer<void, TagA> pv(static_cast<void*>(&x));
    assert(pv.get() == static_cast<void*>(&x));
    pv = pa - 1;  // any typed pointer can be assigned to strong void pointer with the same tag
    // pv = pv + 1;  // strong_pointer<void, TagA> warning as ‘void *’ used in arithmetic
    // [-Wpointer-arith] Note: *pv would not compile, which is correct behavior

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
    Base*                         _x = &d;
    Derived*                      _y = static_cast<Derived*>(_x);
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


// Test fixture for raw pointer-related tests
class PointerTest : public ::testing::Test {
protected:
    int value    = 10;
    int array[5] = {1, 2, 3, 4, 5};

    // Base and Derived classes for inheritance tests
    class Base {
    public:
        virtual ~Base() = default;  // Polymorphic for dynamic_cast
        int base_value  = 42;
    };

    class Derived : public Base {
    public:
        int derived_value = 84;
    };

//#define RAW_POINTER_TESTS
#ifdef RAW_POINTER_TESTS
    using int_p       = int*;
    using const_int_p = const int*;
    using void_p      = void*;
    using char_p      = char*;
    using float_p     = float*;
    using Base_p      = Base*;
    using Derived_p   = Derived*;
    using func_p      = int (*)(int);
#else

    using int_p       = strong_pointer<int, MemoryTag::host>;
    using const_int_p = strong_pointer<const int, MemoryTag::host>;
    using void_p      = strong_pointer<void, MemoryTag::host>;
    using char_p      = strong_pointer<char, MemoryTag::host>;
    using float_p     = strong_pointer<float, MemoryTag::host>;
    using Base_p      = strong_pointer<Base, MemoryTag::host>;
    using Derived_p   = strong_pointer<Derived, MemoryTag::host>;
    using func_p      = int (*)(int);
#endif
};

// 1. Basic Pointer Initialization and Dereferencing
TEST_F(PointerTest, BasicInitializationAndDereference) {
    int_p ptr = &value;
    EXPECT_EQ(*ptr, 10);

    int_p null_ptr = nullptr;
    EXPECT_EQ(null_ptr, nullptr);  // Null pointer check
}

// 2. Interaction with void_p
TEST_F(PointerTest, VoidPointerInteraction) {
    int_p  iptr = &value;
    void_p vptr = iptr;     // Implicit conversion to void_p
    EXPECT_EQ(vptr, iptr);  // Same address

    int_p back_ptr = static_cast<int_p>(vptr);
    EXPECT_EQ(*back_ptr, 10);

    EXPECT_EQ(vptr, iptr);
    EXPECT_EQ(iptr, vptr);

    // Illegal: Cannot dereference void_p
    // *vptr; // Compile error: void_p has no type information
}

// 3. Interaction with nullptr
TEST_F(PointerTest, NullptrInteraction) {
    int_p ptr = nullptr;
    EXPECT_TRUE(ptr == nullptr);
    EXPECT_FALSE(ptr != nullptr);

    void_p vptr = nullptr;
    EXPECT_EQ(vptr, nullptr);

    // Assign nullptr to function pointer
    void (*fptr)() = nullptr;
    EXPECT_EQ(fptr, nullptr);
}

// 4. Casting Between Base and Derived Pointers
TEST_F(PointerTest, BaseDerivedCasting) {
    Derived d;
    Base_p  bptr = &d;  // Implicit upcast
    EXPECT_EQ(bptr->base_value, 42);

    // Downcast with static_cast (assumes correct type)
    Derived_p dptr_static = static_cast<Derived_p>(bptr);
    EXPECT_EQ(dptr_static->derived_value, 84);

#ifdef RAW_POINTER_TESTS
    // Downcast with dynamic_cast (runtime check)
    Derived_p dptr_dynamic = dynamic_cast<Derived_p>(bptr);
    EXPECT_NE(dptr_dynamic, nullptr);
    EXPECT_EQ(dptr_dynamic->derived_value, 84);

    // dynamic_cast failure case
    Base      b;
    Base_p    bptr2     = &b;
    Derived_p dptr_fail = dynamic_cast<Derived_p>(bptr2);
    EXPECT_EQ(dptr_fail, nullptr);

#else  // strong_pointer have different cast call

    // Downcast with dynamic_cast (runtime check)
    Derived_p dptr_dynamic = bptr.cast_dynamic<Derived_p>();
    EXPECT_NE(dptr_dynamic, nullptr);
    EXPECT_EQ(dptr_dynamic->derived_value, 84);

    // dynamic_cast failure case
    Base      b;
    Base_p    bptr2     = &b;
    Derived_p dptr_fail = bptr2.cast_dynamic<Derived_p>();
    EXPECT_EQ(dptr_fail, nullptr);
#endif
}

// 5. Pointer Comparisons
TEST_F(PointerTest, PointerComparisons) {
    int_p p1 = &array[1];
    int_p p2 = &array[3];

    // Relational comparison within same array
    EXPECT_TRUE(p1 < p2);
    EXPECT_TRUE(p2 > p1);
    EXPECT_FALSE(p1 == p2);

    // Equality comparison
    int_p p3 = &array[1];
    EXPECT_TRUE(p1 == p3);

    // Comparison with nullptr
    EXPECT_FALSE(p1 == nullptr);
    EXPECT_TRUE(nullptr == nullptr);

    // Illegal: Comparing pointers to unrelated objects (UB)
    int   x = 10, y = 20;
    int_p px = &x;
    int_p py = &y;
    // EXPECT_TRUE(px < py); // UB: Undefined behavior, commented out
}

// 6. Pointer Arithmetic
TEST_F(PointerTest, PointerArithmetic) {
    int_p ptr = array;
    EXPECT_EQ(*ptr, 1);

    ptr += 2;
    EXPECT_EQ(*ptr, 3);

    ptr--;
    EXPECT_EQ(*ptr, 2);

    ptrdiff_t diff = ptr - array;
    EXPECT_EQ(diff, 1);

    // One-past-the-end is valid
    int_p end = array + 5;
    EXPECT_EQ(end - array, 5);

    // Illegal: Dereferencing one-past-the-end (UB)
    // int val = *end; // UB: Out of bounds, commented out

    // Illegal: Arithmetic beyond one-past-the-end (UB)
    // int_p bad_ptr = array + 6; // UB if dereferenced or used improperly
}

// 7. Const Correctness
TEST_F(PointerTest, ConstCorrectness) {
    const_int_p cptr = &value;  // Pointer to const int
    EXPECT_EQ(*cptr, 10);
    // *cptr = 20; // Illegal: Cannot modify through const pointer

#ifdef RAW_POINTER_TESTS
    *const_cast<int*>(cptr) = 20;
#else
    *cptr.cast_const<int_p>() = 20;
#endif
    EXPECT_EQ(*cptr, 20);

    int_p const ptr_const = &value;  // Const pointer
    *ptr_const            = 15;
    EXPECT_EQ(value, 15);
    // ptr_const = nullptr; // Illegal: Cannot reassign const pointer

    const_int_p const cptr_const = &value;  // Both const
    EXPECT_EQ(*cptr_const, 15);
    // *cptr_const = 20; // Illegal: Cannot modify
    // cptr_const = nullptr; // Illegal: Cannot reassign
}

// 8. Function Pointers
TEST_F(PointerTest, FunctionPointers) {
    auto   func = [](int x) { return x + 1; };
    func_p fptr = func;
    EXPECT_EQ(fptr(5), 6);

    fptr = nullptr;
    EXPECT_EQ(fptr, nullptr);

    // Illegal: Cannot implicitly convert void_p to function pointer
    void_p vptr = nullptr;
    // fptr = static_cast<func_p>(vptr); // Compile error in standard C++
}

// 9. Array-to-Pointer Decay
TEST_F(PointerTest, ArrayDecay) {
    int_p ptr = array;  // Decays to pointer to first element
    EXPECT_EQ(*ptr, 1);

    ptr = &array[0];  // Equivalent
    EXPECT_EQ(*ptr, 1);
}

// 10. Dangling Pointer (Simulated)
TEST_F(PointerTest, DanglingPointer) {
    int_p ptr;
    {
        int temp = 99;
        ptr      = &temp;
        EXPECT_EQ(*ptr, 99);  // Valid here
    }
    // *ptr; // UB: Dangling pointer after temp is destroyed, commented out
}

// 11. Reinterpret_cast
TEST_F(PointerTest, ReinterpretCast) {
    int   x    = 10;
    int_p iptr = &x;

#ifdef RAW_POINTER_TESTS
    float_p fptr = reinterpret_cast<float_p>(iptr);
    //  Dereferencing fptr would be UB due to type mismatch
    EXPECT_EQ(fptr, reinterpret_cast<float_p>(iptr));  // Same address

    // Illegal: Misaligned access (UB if dereferenced)
    char_p cptr = reinterpret_cast<char_p>(&x);
    // int_p misaligned = reinterpret_cast<int_p>(cptr + 1); // UB if used

#else  // strong_pointer have different cast call
    float_p fptr              = iptr.cast_reinterpret<float_p>();
    // Dereferencing fptr would be UB due to type mismatch
    EXPECT_EQ(fptr, iptr.cast_reinterpret<float_p>());  // Same address
                                                        //
    // Illegal: Misaligned access (UB if dereferenced)
    char_p cptr = iptr.cast_reinterpret<char_p>();
    // int_p misaligned = (cptr + 1).cast_reinterpret<int_p>(); // UB if used
#endif
}

// 12. Undefined Behavior Cases (Commented Out)
TEST_F(PointerTest, UndefinedBehaviorCases) {
    int_p ptr = nullptr;
    // *ptr; // UB: Dereferencing nullptr, commented out

    int   arr[2];
    int_p out_of_bounds = arr + 3;
    // *out_of_bounds; // UB: Dereferencing out of bounds, commented out

    int   x, y;
    int_p px = &x;
    int_p py = &y;
    // bool cmp = (px < py); // UB: Comparing unrelated pointers, commented out
}

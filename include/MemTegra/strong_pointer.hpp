#ifndef STRONG_POINTER_H
#define STRONG_POINTER_H

#include <cstddef>
#include <type_traits>

#include "./MemTegra.h"

namespace MT {
    template <typename T, typename Tag> class strong_pointer {
    public:
        using value_type = T;
        using tag_type   = Tag;

        // Constructor
        explicit strong_pointer(T* ptr = nullptr) : ptr_(ptr) {}

        // Access underlying pointer
        T* get() const { return ptr_; }

        // Dereference operators
        T& operator*() const {
            static_assert(internal::support_reference<Tag>::value,
                          "Memory do not support reference");
            return *ptr_;
        }
        T* operator->() const {
            static_assert(internal::support_reference<Tag>::value,
                          "Memory do not support reference");
            return ptr_;
        }

        // Indexing operator
        T& operator[](std::size_t index) const { return ptr_[index]; }

        // Arithmetic operations
        strong_pointer operator+(std::ptrdiff_t offset) const {
            return strong_pointer(ptr_ + offset);
        }

        strong_pointer& operator+=(std::ptrdiff_t offset) {
            ptr_ += offset;
            return *this;
        }

        strong_pointer operator-(std::ptrdiff_t offset) const {
            return strong_pointer(ptr_ - offset);
        }

        strong_pointer& operator-=(std::ptrdiff_t offset) {
            ptr_ -= offset;
            return *this;
        }

        std::ptrdiff_t operator-(const strong_pointer& other) const { return ptr_ - other.ptr_; }

        // Comparison operators
        bool operator==(const strong_pointer& other) const { return ptr_ == other.ptr_; }

        bool operator!=(const strong_pointer& other) const { return !(*this == other); }

        bool operator<(const strong_pointer& other) const { return ptr_ < other.ptr_; }

        bool operator<=(const strong_pointer& other) const { return !(*this > other); }

        bool operator>(const strong_pointer& other) const { return other < *this; }

        bool operator>=(const strong_pointer& other) const { return !(*this < other); }

    private:
        T* ptr_;
    };

    using int_hp  = strong_pointer<int, MemoryTag::ENUM_HOST>;
    using void_hp = strong_pointer<void, MemoryTag::ENUM_HOST>;
};  // namespace MT

#endif  // STRONG_POINTER_H

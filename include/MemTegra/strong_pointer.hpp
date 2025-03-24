#ifndef STRONG_POINTER_H
#define STRONG_POINTER_H

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "./MemTegra.h"

namespace MT {
    template <typename T, typename Tag> class strong_pointer {
        // dereference_type used here to bypass compiler error for void type
        using dereference_type = typename std::conditional<std::is_void_v<T>, int, T>::type;

    public:
        using value_type = T;
        using tag_type   = Tag;

        // Constructor
        explicit strong_pointer(T* ptr = nullptr) : ptr_(ptr) {}
        explicit strong_pointer(const strong_pointer<void, Tag>& ptr)
            : ptr_(static_cast<T*>(ptr.get())) {}

        // Conversion operator to strong_pointer<void, Tag>
        operator strong_pointer<void, Tag>() const { return strong_pointer<void, Tag>(ptr_); }


        // Access underlying pointer
        T* get() const { return ptr_; }

        // Dereference operators
        dereference_type& operator*() const {
            static_assert(internal::support_reference<T, Tag>::value,
                          "Memory do not support reference");
            return *static_cast<dereference_type*>(ptr_);
        }
        dereference_type& operator->() const {
            static_assert(internal::support_reference<T, Tag>::value,
                          "Memory do not support reference");
            return *static_cast<dereference_type*>(ptr_);
        }

        // Indexing operator
        dereference_type& operator[](std::size_t index) const {
            static_assert(internal::support_reference<T, Tag>::value,
                          "Memory do not support reference");
            return static_cast<dereference_type*>(ptr_)[index];
        }

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
        bool operator==(const std::nullptr_t& other) const { return ptr_ == other; }
        bool operator!=(const std::nullptr_t& other) const { return !(*this == other); }

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
};      // namespace MT
#endif  // STRONG_POINTER_H

#ifndef STRONG_POINTER_H
#define STRONG_POINTER_H

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "./memory_tags.h"

namespace MT {
    namespace internal {
        template <typename T, typename Tag> struct support_reference {
            constexpr static bool value = true;
        };
        template <typename Tag> struct support_reference<void, Tag> {
            constexpr static bool value = false;
        };
    };  // namespace internal
        //
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
        operator T*() const { return get(); }


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
};  // namespace MT


    // namespace MT {
    //// Memory set operation for strong pointers
    // template <typename Tag>
    // strong_pointer<void, Tag> memset(strong_pointer<void, Tag>& ptr, int ch, std::size_t count) {
    // if (!ptr) {
    // throw std::runtime_error("Null strong pointer.");
    //}
    // std::memset(ptr.get(), ch, count);
    // return ptr;
    //}

    //// Memory copy operation for strong pointers
    // template <typename T, typename Tag1, typename Tag2>
    // void memcpy(const strong_pointer<T, Tag1>& src, strong_pointer<T, Tag2>& dest) {
    // if (!src) {
    // throw std::runtime_error("Source strong pointer is null.");
    //}
    // if (!dest) {
    // throw std::runtime_error("Destination strong pointer is null.");
    //}

    // if constexpr (std::is_same<Tag1, MemoryTag::ENUM_HOST>::value && std::is_same<Tag2,
    // MemoryTag::ENUM_HOST>::value) {
    // std::memcpy(dest.get(), src.get(), sizeof(T));
    //} else if constexpr (std::is_same<Tag1, MemoryTag::ENUM_HOST>::value &&
    // std::is_same<Tag2, MemoryTag::ENUM_DEVICE>::value) {
    // cudaMemcpy(dest.get(), src.get(), sizeof(T), cudaMemcpyHostToDevice);
    //} else {
    // throw std::runtime_error("Unsupported strong pointer types for copy operation.");
    //}
    //}
    //};  // namespace MT

#endif  // STRONG_POINTER_H

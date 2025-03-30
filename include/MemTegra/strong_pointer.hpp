#ifndef STRONG_POINTER_H
#define STRONG_POINTER_H

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "./memory_tags.h"

namespace MT {
    template <typename T> struct strong_pointer_traits {
        constexpr static bool support_reference = true;
    };

    template <typename T, typename Tag> class strong_pointer {
        using dereference_type = typename std::conditional_t<std::is_void_v<T>, char, T>&;

    public:
        // **Iterator Traits** //
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = T*;
        using reference         = std::conditional_t<std::is_void_v<T>, void, dereference_type>;

        // **Constructors** //
        strong_pointer() noexcept : ptr(nullptr) {}

        // Constructor from raw pointer
        /*explicit*/ strong_pointer(T* p) noexcept : ptr(p) {}

        // Conversion constructor from another strong_pointer with the same tag
        template <typename U = T, typename Tag2 = Tag,
                  typename = std::enable_if_t<
                      std::is_convertible_v<U*, T*> && std::is_convertible_v<Tag2*, Tag*>>>
        /*explicit*/ strong_pointer(const strong_pointer<U, Tag2>& other) noexcept
            : ptr(other.get()) {}

        template <typename U = T, typename Tag2 = Tag,
                  typename = std::enable_if_t<
                      std::is_convertible_v<U*, T*> && std::is_convertible_v<Tag2*, Tag*>>>
        /*explicit*/ strong_pointer(strong_pointer<U, Tag2>&& other) noexcept : ptr(other.get()) {}

        // convertible to another strong_pointer with the same tag but must be explicit
        template <typename U = T, typename = std::enable_if_t<!std::is_convertible_v<T*, U*>>>
        explicit operator strong_pointer<U, Tag>() const noexcept {
            return strong_pointer<U, Tag>(static_cast<U*>(ptr));
        }

        // **Accessors** //
        T* get() const noexcept { return ptr; }

        template <typename U = T>
        std::enable_if_t<!std::is_void_v<U>, U>& operator*() const noexcept {
            static_assert(strong_pointer_traits<strong_pointer<U, Tag>>::support_reference,
                          "Memory do not support reference");
            return *ptr;
        }

        template <typename U = T>
        std::enable_if_t<!std::is_void_v<U>, U>* operator->() const noexcept {
            static_assert(strong_pointer_traits<strong_pointer<U, Tag>>::support_reference,
                          "Memory do not support reference");
            return ptr;
        }

        template <typename U = T>
        std::enable_if_t<!std::is_void_v<U>, U>& operator[](std::size_t index) const noexcept {
            static_assert(strong_pointer_traits<strong_pointer<U, Tag>>::support_reference,
                          "Memory do not support reference");
            return ptr[index];
        }

        // **Pointer Arithmetic** //
        strong_pointer operator+(std::ptrdiff_t n) const noexcept {
            return strong_pointer(ptr + n);
        }

        strong_pointer operator-(std::ptrdiff_t n) const noexcept {
            return strong_pointer(ptr - n);
        }

        strong_pointer& operator+=(std::ptrdiff_t n) noexcept {
            ptr += n;
            return *this;
        }

        strong_pointer& operator-=(std::ptrdiff_t n) noexcept {
            ptr -= n;
            return *this;
        }

        strong_pointer& operator++() noexcept {
            ++ptr;
            return *this;
        }

        strong_pointer operator++(int) noexcept {
            strong_pointer temp(*this);
            ++(*this);
            return temp;
        }

        strong_pointer& operator--() noexcept {
            --ptr;
            return *this;
        }

        strong_pointer operator--(int) noexcept {
            strong_pointer temp(*this);
            --(*this);
            return temp;
        }

        std::ptrdiff_t operator-(const strong_pointer& other) const noexcept {
            return ptr - other.ptr;
        }

        // **Comparison Operators** //
        bool operator==(const strong_pointer& other) const noexcept { return ptr == other.get(); }
        bool operator!=(const strong_pointer& other) const noexcept { return ptr != other.get(); }
        bool operator>(const strong_pointer& other) const noexcept { return ptr > other.get(); }
        bool operator<(const strong_pointer& other) const noexcept { return ptr < other.get(); }
        bool operator>=(const strong_pointer& other) const noexcept { return ptr >= other.get(); }
        bool operator<=(const strong_pointer& other) const noexcept { return ptr <= other.get(); }
        bool operator==(std::nullptr_t) const noexcept { return ptr == nullptr; }
        bool operator!=(std::nullptr_t) const noexcept { return ptr != nullptr; }

        // **Assignment** //
        strong_pointer& operator=(const strong_pointer& other) noexcept {
            ptr = other.get();
            return *this;
        }

        strong_pointer& operator=(T* p) noexcept {
            ptr = p;
            return *this;
        }

        strong_pointer& operator=(std::nullptr_t) noexcept {
            ptr = nullptr;
            return *this;
        }

        // **Miscellaneous**
        explicit operator bool() const noexcept { return ptr != nullptr; }

    private:
        T* ptr;  // The underlying raw pointer
    };

    using int_hp  = strong_pointer<int, MemoryTag::host>;
    using void_hp = strong_pointer<void, MemoryTag::host>;

    // This specialization actually do not needed for strong_pointer
    template <typename Tag> struct strong_pointer_traits<strong_pointer<void, Tag>> {
        constexpr static bool support_reference = false;
    };

};  // namespace MT

namespace std {
    // TODO: fix reference for void and add const specialization, maybe move into class
    template <typename T, typename Tag> struct iterator_traits<MT::strong_pointer<T, Tag>> {
        using difference_type   = typename iterator_traits<T*>::difference_type;
        using value_type        = typename iterator_traits<T*>::value_type;
        using pointer           = typename iterator_traits<T*>::pointer;
        using reference         = typename iterator_traits<T*>::reference;
        using iterator_category = typename iterator_traits<T*>::iterator_category;
    };
}  // namespace std


#endif  // STRONG_POINTER_H

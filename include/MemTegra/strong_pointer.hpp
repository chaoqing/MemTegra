#ifndef STRONG_POINTER_H
#define STRONG_POINTER_H

#include <cstddef>
#include <cstring>
#include <iostream>
#include <type_traits>

#include "./memory_tags.h"

namespace MT {
    template <typename T> struct strong_pointer_traits {
        constexpr static bool support_reference = true;
        constexpr static bool is_strong_pointer = false;
        using memory_tag                        = void;
        using value_type                        = void;
    };

    template <typename T, typename Tag> class strong_pointer {
        using dereference_type = typename std::conditional_t<std::is_void_v<T>, char, T>&;
        static_assert(!std::is_same_v<Tag, void>, "strong_pointer memory_tag should not be void");

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
        template <typename U = T, typename = std::enable_if_t<!std::is_void_v<U>, bool>>
        bool operator==(const strong_pointer<void, Tag>& other) const noexcept {
            return ptr == other.get();
        }
        template <typename U = T, typename = std::enable_if_t<!std::is_void_v<U>, bool>>
        bool operator!=(const strong_pointer<void, Tag>& other) const noexcept {
            return ptr != other.get();
        }
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


        template <typename U,
                  typename = std::enable_if_t<
                      strong_pointer_traits<U>::is_strong_pointer
                          && std::is_same_v<Tag, typename strong_pointer_traits<U>::memory_tag>,
                      U>>
        U cast_const() const noexcept {
            using UT = typename strong_pointer_traits<U>::value_type;
            return const_cast<UT*>(ptr);
        }
        template <typename U,
                  typename = std::enable_if_t<
                      strong_pointer_traits<U>::is_strong_pointer
                          && std::is_same_v<Tag, typename strong_pointer_traits<U>::memory_tag>,
                      U>>
        U cast_static() const noexcept {
            using UT = typename strong_pointer_traits<U>::value_type;
            return static_cast<UT*>(ptr);
        }

        template <typename U,
                  typename = std::enable_if_t<
                      strong_pointer_traits<U>::is_strong_pointer
                          && std::is_same_v<Tag, typename strong_pointer_traits<U>::memory_tag>,
                      U>>
        U cast_dynamic() const noexcept {
            using UT = typename strong_pointer_traits<U>::value_type;
            return dynamic_cast<UT*>(ptr);
        }
        template <typename U,
                  typename = std::enable_if_t<
                      strong_pointer_traits<U>::is_strong_pointer
                          && std::is_same_v<Tag, typename strong_pointer_traits<U>::memory_tag>,
                      U>>
        U cast_reinterpret() const noexcept {
            using UT = typename strong_pointer_traits<U>::value_type;
            return reinterpret_cast<UT*>(ptr);
        }

    private:
        T* ptr;  // The underlying raw pointer
    };

    // equality with raw pointers
    template <typename T, typename Tag, typename = std::enable_if_t<!std::is_same_v<T, void>>>
    bool operator==(void* lhs, const strong_pointer<T, Tag>& ohs) noexcept {
        return lhs == ohs.get();
    }
    template <typename T, typename Tag, typename = std::enable_if_t<!std::is_same_v<T, void>>>
    bool operator!=(void* lhs, const strong_pointer<T, Tag>& ohs) noexcept {
        return lhs != ohs.get();
    }

    template <typename T, typename Tag>
    bool operator==(T* lhs, const strong_pointer<T, Tag>& ohs) noexcept {
        return lhs == ohs.get();
    }
    template <typename T, typename Tag>
    bool operator!=(T* lhs, const strong_pointer<T, Tag>& ohs) noexcept {
        return lhs != ohs.get();
    }

    // I/O
    template <typename T, typename Tag>
    std::ostream& operator<<(std::ostream& os, const strong_pointer<T, Tag>& sp) {
        os << sp.get();
        return os;
    }

    template <typename T, typename Tag>
    std::istream& operator>>(std::istream& is, strong_pointer<T, Tag>& sp) {
        void* raw_ptr;
        is >> raw_ptr;
        sp = static_cast<T*>(raw_ptr);
        return is;
    }

    template <typename T, typename Tag> struct strong_pointer_traits<strong_pointer<T, Tag>> {
        constexpr static bool support_reference = true;
        constexpr static bool is_strong_pointer = true;
        using memory_tag                        = Tag;
        using value_type                        = T;
    };

    // This specialization actually do not needed for strong_pointer
    template <typename Tag> struct strong_pointer_traits<strong_pointer<void, Tag>> {
        constexpr static bool support_reference = false;
        constexpr static bool is_strong_pointer = true;
        using memory_tag                        = Tag;
        using value_type                        = void;
    };

    // Common types
    using int_hp  = strong_pointer<int, MemoryTag::host>;
    using void_hp = strong_pointer<void, MemoryTag::host>;

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

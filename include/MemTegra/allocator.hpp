#ifndef MEMTEGRA_ALLOCATOR_HPP
#define MEMTEGRA_ALLOCATOR_HPP

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>

#include "./strong_pointer.hpp"

namespace MT {
    template <typename Tag> class RawAllocator {
    public:
        static void* malloc(std::size_t size) {
            if (size == 0) {
                throw std::invalid_argument("Size must be greater than zero.");
            }

            void* ptr = std::malloc(size);
            if (ptr == 0) {
                throw std::runtime_error("Memory allocation failed.");
            }
            return ptr;
        }

        static void free(void* ptr) {
            if (ptr != nullptr) {
                std::free(ptr);
            }
        }
    };

    // Specialization for host_aligned_64
    template <size_t N> class RawAllocator<MemoryTag::host_aligned<N>> {
    public:
        static void* malloc(std::size_t size) {
            if (size == 0) {
                throw std::invalid_argument("Size must be greater than zero.");
            }

            constexpr std::size_t alignment = MemoryTag::host_aligned<N>::alignment;
            void*                 ptr
                = std::aligned_alloc(alignment, ((size + alignment - 1) / alignment) * alignment);
            if (ptr == nullptr) {
                throw std::runtime_error("Aligned memory allocation failed.");
            }
            return ptr;
        }

        static void free(void* ptr) {
            if (ptr != nullptr) {
                std::free(ptr);
            }
        }
    };

};  // namespace MT


namespace MT {
    template <typename T, typename Tag> class MemTegraAllocator {
        using raw_allocator = RawAllocator<Tag>;

    public:
        using value_type = T;
        using pointer    = strong_pointer<T, Tag>;

        MemTegraAllocator() noexcept = default;

        template <typename U> MemTegraAllocator(const MemTegraAllocator<U, Tag>&) noexcept {}

        pointer allocate(std::size_t n) {
            if (n == 0) {
                return nullptr;
            }
            if (n > static_cast<std::size_t>(-1) / sizeof(T)) {
                throw std::bad_alloc();
            }
            return static_cast<T*>(raw_allocator::malloc(n * sizeof(T)));
        }

        void deallocate(const pointer& p, std::size_t) { raw_allocator::free(p.get()); }

        template <typename U, typename Tag2>
        bool operator==(const MemTegraAllocator<U, Tag2>&) const noexcept {
            return std::is_same_v<Tag, Tag2>;
        }

        template <typename U, typename Tag2>
        bool operator!=(const MemTegraAllocator<U, Tag2>& ohs) const noexcept {
            return !(*this == ohs);
        }
    };

};      // namespace MT
#endif  // MEMTEGRA_ALLOCATOR_HPP

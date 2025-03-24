// MemTegra.h
#ifndef MEMTEGRA_H
#define MEMTEGRA_H

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
namespace MT {
    namespace MemoryTag {
        enum class ENUM_HOST {};
        enum class ENUM_ALIGNED_64 {};
    };  // namespace MemoryTag

    namespace internal {
        template <typename Tag> struct support_reference { constexpr static bool value = true; };
    };  // namespace internal
};      // namespace MT


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

    // Specialization for ENUM_ALIGNED_64
    template <> class RawAllocator<MemoryTag::ENUM_ALIGNED_64> {
    public:
        static void* malloc(std::size_t size) {
            if (size == 0) {
                throw std::invalid_argument("Size must be greater than zero.");
            }

            constexpr std::size_t alignment = 64;
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

        MemTegraAllocator() noexcept = default;

        template <typename U, typename Tag2>
        MemTegraAllocator(const MemTegraAllocator<U, Tag2>&) noexcept {}

        T* allocate(std::size_t n) {
            if (n == 0) {
                return nullptr;
            }
            if (n > static_cast<std::size_t>(-1) / sizeof(T)) {
                throw std::bad_alloc();
            }
            return static_cast<T*>(raw_allocator::malloc(n * sizeof(T)));
        }

        void deallocate(T* p, std::size_t) noexcept { raw_allocator::free(p); }

        template <typename U, typename Tag2>
        bool operator==(const MemTegraAllocator<U, Tag2>&) const noexcept {
            return std::is_same<Tag, Tag2>::value;
        }

        template <typename U, typename Tag2>
        bool operator!=(const MemTegraAllocator<U, Tag2>& ohs) const noexcept {
            return !(*this == ohs);
        }
    };

};  // namespace MT


#endif  // MEMTEGRA_#ifndef MEMTEGRA_ALLOCATOR_H
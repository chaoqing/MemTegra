#ifndef MEMTEGRA_DEVICE_ALLOCATOR_H
#define MEMTEGRA_DEVICE_ALLOCATOR_H

#include <type_traits>

#include "./MemTegra.h"
#include "./strong_pointer.hpp"
namespace MT {

    namespace MemoryTag {
        struct device {};
    };  // namespace MemoryTag
    template <typename T> struct strong_pointer_traits<strong_pointer<T, MemoryTag::device>> {
        constexpr static bool support_reference = false;
    };

    namespace internal {
        void* cuda_malloc(std::size_t);
        void  cuda_free(void*);
    };  // namespace internal

    // Specialization for device
    template <> class RawAllocator<MemoryTag::device> {
    public:
        static void* malloc(std::size_t size) { return internal::cuda_malloc(size); }

        static void free(void* ptr) { internal::cuda_free(ptr); }
    };

    using int_dp  = strong_pointer<int, MemoryTag::device>;
    using void_dp = strong_pointer<void, MemoryTag::device>;
};  // namespace MT


namespace MT {
    namespace internal {
        using cudaStream_t = void*;

        void* cuda_memset(void* dest, int ch, size_t n, cudaStream_t stream = nullptr);

        enum class cudaMemcpyKind {
            cudaMemcpyHostToDevice   = 1, /**< Host   -> Device */
            cudaMemcpyDeviceToHost   = 2, /**< Device -> Host */
            cudaMemcpyDeviceToDevice = 3, /**< Device -> Device */
        };

        void* cuda_memcpy(void* dest, const void* src, size_t n, cudaMemcpyKind kind,
                          cudaStream_t stream = nullptr);
    };  // namespace internal

    class cuda_context {
    public:
        cuda_context(internal::cudaStream_t stream = nullptr) : stream_(stream) {}

        // Memory set operation for strong pointers
        template <typename T, typename Tag>
        strong_pointer<T, Tag> memset(const strong_pointer<T, Tag>& ptr, int ch,
                                      std::size_t count) const {
            if (!ptr) {
                throw std::runtime_error("Null strong pointer.");
            }
            if constexpr (std::is_convertible_v<Tag*, MemoryTag::device*>) {
                internal::cuda_memset(ptr.get(), ch, count, stream_);
            } else {
                std::memset(ptr.get(), ch, count);
            }
            return ptr;
        }

        // Memory copy operation for strong pointers
        template <typename T, typename U, typename DestTag, typename SrcTag>
        strong_pointer<T, DestTag> memcpy(const strong_pointer<T, DestTag>& dest,
                                          const strong_pointer<U, SrcTag>& src, size_t n) {
            if (!src) {
                throw std::runtime_error("Source strong pointer is null.");
            }
            if (!dest) {
                throw std::runtime_error("Destination strong pointer is null.");
            }

            if constexpr (std::is_convertible_v<
                              SrcTag*,
                              MemoryTag::
                                  device*> && std::is_convertible_v<DestTag*, MemoryTag::device*>) {
                internal::cuda_memcpy(dest.get(), src.get(), n,
                                      internal::cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream_);
            } else if constexpr (
                std::is_convertible_v<
                    SrcTag*,
                    MemoryTag::device*> && !std::is_convertible_v<DestTag*, MemoryTag::device*>) {
                internal::cuda_memcpy(dest.get(), src.get(), n,
                                      internal::cudaMemcpyKind::cudaMemcpyDeviceToHost, stream_);
            } else if constexpr (
                !std::is_convertible_v<
                    SrcTag*,
                    MemoryTag::device*> && std::is_convertible_v<DestTag*, MemoryTag::device*>) {
                internal::cuda_memcpy(dest.get(), src.get(), n,
                                      internal::cudaMemcpyKind::cudaMemcpyHostToDevice, stream_);
            } else {
                std::memcpy(dest.get(), src.get(), n);
            }
            return dest;
        }

    private:
        const internal::cudaStream_t stream_;
    };
};      // namespace MT
#endif  // MEMTEGRA_DEVICE_ALLOCATOR_H

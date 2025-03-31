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
        constexpr static bool is_strong_pointer = true;
        using memory_tag                        = MemoryTag::device;
        using value_type                        = T;
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
    namespace cuda {
        bool is_availale();

        class context {
            using cudaStream_t = void*;

            enum class cudaMemcpyKind {
                cudaMemcpyHostToHost     = 0,
                cudaMemcpyHostToDevice   = 1,
                cudaMemcpyDeviceToHost   = 2,
                cudaMemcpyDeviceToDevice = 3,
            };

        public:
            context() : stream_(nullptr), async_(false) {}
            context(cudaStream_t stream) : stream_(stream), async_(true) {}
            context(const context&) = default;
            context(context&&)      = default;

            // Memory set operation for strong pointers
            template <typename T, typename Tag>
            strong_pointer<T, Tag> memset(const strong_pointer<T, Tag>& ptr, int ch,
                                          std::size_t count) const {
                if (!ptr) {
                    throw std::runtime_error("Null strong pointer.");
                }
                if constexpr (std::is_convertible_v<Tag*, MemoryTag::device*>) {
                    cuda_memset(ptr.get(), ch, count);
                } else {
                    std::memset(ptr.get(), ch, count);
                }
                return ptr;
            }

            // Memory copy operation for strong pointers
            template <typename T, typename U, typename DestTag, typename SrcTag>
            strong_pointer<T, DestTag> memcpy(const strong_pointer<T, DestTag>& dest,
                                              const strong_pointer<U, SrcTag>&  src,
                                              size_t                            n) const {
                if (!src) {
                    throw std::runtime_error("Source strong pointer is null.");
                }
                if (!dest) {
                    throw std::runtime_error("Destination strong pointer is null.");
                }

                if constexpr (
                    std::is_convertible_v<
                        SrcTag*,
                        MemoryTag::
                            device*> && std::is_convertible_v<DestTag*, MemoryTag::device*>) {
                    cuda_memcpy(dest.get(), src.get(), n, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
                } else if constexpr (
                    std::is_convertible_v<
                        SrcTag*,
                        MemoryTag::
                            device*> && !std::is_convertible_v<DestTag*, MemoryTag::device*>) {
                    cuda_memcpy(dest.get(), src.get(), n, cudaMemcpyKind::cudaMemcpyDeviceToHost);
                } else if constexpr (
                    !std::is_convertible_v<
                        SrcTag*,
                        MemoryTag::
                            device*> && std::is_convertible_v<DestTag*, MemoryTag::device*>) {
                    cuda_memcpy(dest.get(), src.get(), n, cudaMemcpyKind::cudaMemcpyHostToDevice);
                } else {
                    cuda_memcpy(dest.get(), src.get(), n, cudaMemcpyKind::cudaMemcpyHostToHost);
                }
                return dest;
            }

        private:
            const cudaStream_t stream_;
            const bool         async_;

            void* cuda_memset(void* dest, int ch, size_t n) const;
            void* cuda_memcpy(void* dest, const void* src, size_t n, cudaMemcpyKind kind) const;
        };

    };  // namespace cuda
};      // namespace MT
#endif  // MEMTEGRA_DEVICE_ALLOCATOR_H

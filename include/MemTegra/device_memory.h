#ifndef MEMTEGRA_DEVICE_ALLOCATOR_H
#define MEMTEGRA_DEVICE_ALLOCATOR_H


#include "./MemTegra.h"
#include "./strong_pointer.hpp"
namespace MT {

    namespace MemoryTag {
        enum class ENUM_DEVICE {};
    };  // namespace MemoryTag

    namespace internal {
        template <typename T> struct support_reference<T, MemoryTag::ENUM_DEVICE> {
            constexpr static bool value = false;
        };

        void* cuda_malloc(std::size_t);
        void  cuda_free(void*);
    };  // namespace internal

    // Specialization for ENUM_DEVICE
    template <> class RawAllocator<MemoryTag::ENUM_DEVICE> {
    public:
        static void* malloc(std::size_t size) { return internal::cuda_malloc(size); }

        static void free(void* ptr) { internal::cuda_free(ptr); }
    };

    using int_dp  = strong_pointer<int, MemoryTag::ENUM_DEVICE>;
    using void_dp = strong_pointer<void, MemoryTag::ENUM_DEVICE>;
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

    // Memory set operation for strong pointers
    class memset {
    public:
        memset(internal::cudaStream_t stream = nullptr) : stream_(stream) {}
        template <typename T, typename Tag>
        strong_pointer<T, Tag> operator()(const strong_pointer<T, Tag>& ptr, int ch,
                                          std::size_t count) const {
            if (!ptr) {
                throw std::runtime_error("Null strong pointer.");
            }
            std::memset(ptr.get(), ch, count);
            return ptr;
        }

        template <typename T> strong_pointer<T, MemoryTag::ENUM_DEVICE> operator()(
            const strong_pointer<T, MemoryTag::ENUM_DEVICE>& ptr, int ch, std::size_t count) const {
            if (!ptr) {
                throw std::runtime_error("Null strong pointer.");
            }
            internal::cuda_memset(ptr.get(), ch, count, stream_);
            return ptr;
        }

    private:
        internal::cudaStream_t stream_;
    };

    // Memory copy operation for strong pointers
    class memcpy {
    public:
        memcpy(internal::cudaStream_t stream = nullptr) : stream_(stream) {}
        template <typename T, typename U, typename DestTag, typename SrcTag>
        strong_pointer<T, DestTag> operator()(const strong_pointer<T, DestTag>& dest,
                                              const strong_pointer<U, SrcTag>& src, size_t n) {
            if (!src) {
                throw std::runtime_error("Source strong pointer is null.");
            }
            if (!dest) {
                throw std::runtime_error("Destination strong pointer is null.");
            }

            if constexpr (std::is_same_v<
                              SrcTag,
                              MemoryTag::
                                  ENUM_DEVICE> && std::is_same_v<DestTag, MemoryTag::ENUM_DEVICE>) {
                internal::cuda_memcpy(dest.get(), src.get(), n,
                                      internal::cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream_);
            } else if constexpr (
                std::is_same_v<
                    SrcTag,
                    MemoryTag::ENUM_DEVICE> && !std::is_same_v<DestTag, MemoryTag::ENUM_DEVICE>) {
                internal::cuda_memcpy(dest.get(), src.get(), n,
                                      internal::cudaMemcpyKind::cudaMemcpyDeviceToHost, stream_);
            } else if constexpr (
                !std::is_same_v<
                    SrcTag,
                    MemoryTag::ENUM_DEVICE> && std::is_same_v<DestTag, MemoryTag::ENUM_DEVICE>) {
                internal::cuda_memcpy(dest.get(), src.get(), n,
                                      internal::cudaMemcpyKind::cudaMemcpyHostToDevice, stream_);
            } else {
                std::memcpy(dest.get(), src.get(), n);
            }
            return dest;
        }

    private:
        void* stream_;
    };
};      // namespace MT
#endif  // MEMTEGRA_DEVICE_ALLOCATOR_H

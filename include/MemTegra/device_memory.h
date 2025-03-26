#ifndef MEMTEGRA_DEVICE_ALLOCATOR_H
#define MEMTEGRA_DEVICE_ALLOCATOR_H

#ifdef ENABLE_CUDA

#    include "./MemTegra.h"
#    include "./memory_ops.hpp"
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

        void* cudaMemset(void* dest, int ch, size_t n, cudaStream_t stream = nullptr);

        enum class cudaMemcpyKind {
            cudaMemcpyHostToDevice   = 1, /**< Host   -> Device */
            cudaMemcpyDeviceToHost   = 2, /**< Device -> Host */
            cudaMemcpyDeviceToDevice = 3, /**< Device -> Device */
        };

        void* cudaMemcpy(void* dest, const void* src, size_t n, cudaMemcpyKind kind,
                         cudaStream_t stream = nullptr);
    };  // namespace internal

    // Memory set operation for strong pointers
    template <> class memset<MemoryTag::ENUM_DEVICE> {
        using pointer = strong_pointer<void, MemoryTag::ENUM_DEVICE>;

    public:
        memset(internal::cudaStream_t stream = nullptr) : stream_(stream) {}
        pointer operator()(pointer& ptr, int ch, std::size_t count) const {
            if (!ptr) {
                throw std::runtime_error("Null strong pointer.");
            }
            internal::cudaMemset(ptr.get(), ch, count, stream_);
            return ptr;
        }

    private:
        internal::cudaStream_t stream_;
    };

    // Memory copy operation for strong pointers
    template <typename SrcTag> class memcpy<MemoryTag::ENUM_DEVICE, SrcTag> {
    public:
        memcpy(internal::cudaStream_t stream = nullptr) : stream_(stream) {}
        strong_pointer<void, MemoryTag::ENUM_DEVICE> operator()(
            strong_pointer<void, MemoryTag::ENUM_DEVICE>& dest,
            const strong_pointer<void, SrcTag>& src, size_t n) {
            if (!src) {
                throw std::runtime_error("Source strong pointer is null.");
            }
            if (!dest) {
                throw std::runtime_error("Destination strong pointer is null.");
            }

            constexpr internal::cudaMemcpyKind kind
                = (std::is_same_v<SrcTag, MemoryTag::ENUM_DEVICE>)
                      ? internal::cudaMemcpyKind::cudaMemcpyDeviceToDevice
                      : internal::cudaMemcpyKind::cudaMemcpyHostToDevice;

            internal::cudaMemcpy(dest.get(), src.get(), n, kind, stream_);

            return dest;
        }

    private:
        void* stream_;
    };

    template <typename DestTag> class memcpy<DestTag, MemoryTag::ENUM_DEVICE> {
    public:
        memcpy(internal::cudaStream_t stream = nullptr) : stream_(stream) {}
        strong_pointer<void, DestTag> operator()(
            strong_pointer<void, DestTag>&                      dest,
            const strong_pointer<void, MemoryTag::ENUM_DEVICE>& src, size_t n) {
            if (!src) {
                throw std::runtime_error("Source strong pointer is null.");
            }
            if (!dest) {
                throw std::runtime_error("Destination strong pointer is null.");
            }

            internal::cudaMemcpy(dest.get(), src.get(), n,
                                 internal::cudaMemcpyKind::cudaMemcpyDeviceToHost, stream_);

            return dest;
        }

    private:
        void* stream_;
    };
};  // namespace MT
#endif
#endif  // MEMTEGRA_DEVICE_ALLOCATOR_H

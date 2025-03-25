#ifndef MEMTEGRA_DEVICE_ALLOCATOR_H
#define MEMTEGRA_DEVICE_ALLOCATOR_H

#ifdef ENABLE_CUDA

#    include "./MemTegra.h"
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

#endif
#endif  // MEMTEGRA_DEVICE_ALLOCATOR_H

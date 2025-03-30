#ifndef MEMTEGRA_MEMORY_TAGS_H
#define MEMTEGRA_MEMORY_TAGS_H

namespace MT {
    namespace MemoryTag {
        struct host {};
        template <size_t N> struct host_aligned : public host {
            static constexpr size_t alignment = N;
        };
        using host_aligned_64  = host_aligned<64>;
        using host_aligned_128 = host_aligned<128>;
        using host_aligned_256 = host_aligned<256>;
        using host_aligned_512 = host_aligned<512>;
    };  // namespace MemoryTag
};      // namespace MT

#endif  // MEMTEGRA_MEMORY_TAGS_H

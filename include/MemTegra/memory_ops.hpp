#ifndef MEMTEGRA_MEMORY_OPS_HPP
#define MEMTEGRA_MEMORY_OPS_HPP

#include <stdexcept>

#include "./strong_pointer.hpp"

namespace MT {
    // Memory set operation for strong pointers
    template <typename Tag> class memset {
    public:
        memset(void* stream = nullptr) : stream_(stream) {}
        strong_pointer<void, Tag> operator()(strong_pointer<void, Tag>& ptr, int ch,
                                             std::size_t count) const {
            if (!ptr) {
                throw std::runtime_error("Null strong pointer.");
            }
            std::memset(ptr.get(), ch, count);
            return ptr;
        }

    private:
        void* stream_;
    };

    // Memory copy operation for strong pointers
    template <typename DestTag, typename SrcTag> class memcpy {
    public:
        memcpy(void* stream = nullptr) : stream_(stream) {}
        strong_pointer<void, DestTag> operator()(strong_pointer<void, DestTag>&      dest,
                                                 const strong_pointer<void, SrcTag>& src,
                                                 size_t                              n) {
            if (!src) {
                throw std::runtime_error("Source strong pointer is null.");
            }
            if (!dest) {
                throw std::runtime_error("Destination strong pointer is null.");
            }

            return std::memcpy(dest.get(), src.get(), n);
        }

    private:
        void* stream_;
    };
};  // namespace MT

#endif  // MEMTEGRA_MEMORY_OPS_HPP

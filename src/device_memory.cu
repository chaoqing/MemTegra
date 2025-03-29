#ifdef ENABLE_CUDA
#    include <cuda_runtime.h>
#else
#    include <cstring>
#endif

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "MemTegra/device_memory.h"

namespace MT {
    namespace internal {
        void* cuda_malloc(std::size_t size) {
            if (size == 0) {
                throw std::invalid_argument("Size must be greater than zero.");
            }

            void* ptr = nullptr;
#ifdef ENABLE_CUDA
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA device memory allocation failed: "
                                         + std::string(cudaGetErrorString(err)));
            }
            return ptr;
#else
            constexpr size_t CUDA_DEFAULT_ALIGNMENT = 256;
            return std::aligned_alloc(size, CUDA_DEFAULT_ALIGNMENT);
#endif
        }

        void cuda_free(void* ptr) {
            if (ptr != nullptr) {
#ifdef ENABLE_CUDA
                cudaError_t err = cudaFree(ptr);
                if (err != cudaSuccess) {
                    throw std::runtime_error("CUDA device memory deallocation failed: "
                                             + std::string(cudaGetErrorString(err)));
                }
#else
                std::free(ptr);
#endif
            }
        }

        void* cuda_memset(void* dest, int ch, size_t n, cudaStream_t stream) {
#ifdef ENABLE_CUDA
            cudaError_t err = cudaSuccess;
            if (stream) {
                err = cudaMemsetAsync(dest, ch, n, static_cast<::cudaStream_t>(stream));
            } else {
                err = cudaMemset(dest, ch, n);
            }
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA device memory memset failed: "
                                         + std::string(cudaGetErrorString(err)));
            }
#else
            std::memset(dest, ch, n);
#endif

            return dest;
        }

        void* cuda_memcpy(void* dest, const void* src, size_t n, cudaMemcpyKind kind,
                          cudaStream_t stream) {
#ifdef ENABLE_CUDA
            cudaError_t err = cudaSuccess;

            static_assert(static_cast<::cudaMemcpyKind>(cudaMemcpyKind::cudaMemcpyHostToDevice)
                          == cudaMemcpyHostToDevice);
            static_assert(static_cast<::cudaMemcpyKind>(cudaMemcpyKind::cudaMemcpyDeviceToHost)
                          == cudaMemcpyDeviceToHost);
            static_assert(static_cast<::cudaMemcpyKind>(cudaMemcpyKind::cudaMemcpyDeviceToDevice)
                          == cudaMemcpyDeviceToDevice);

            const auto _kind = static_cast<::cudaMemcpyKind>(kind);
            if (stream) {
                err = cudaMemcpyAsync(dest, src, n, _kind, static_cast<::cudaStream_t>(stream));
            } else {
                err = cudaMemcpy(dest, src, n, _kind);
            }

            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA device memory memset failed: "
                                         + std::string(cudaGetErrorString(err)));
            }
#else
            std::memcpy(dest, src, n);
#endif

            return dest;
        }

    };  // namespace internal
};      // namespace MT

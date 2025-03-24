#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "MemTegra/device_memory.h"

namespace MT {
    namespace internal {
        void* cuda_malloc(std::size_t size) {
            if (size == 0) {
                throw std::invalid_argument("Size must be greater than zero.");
            }

            void*       ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA device memory allocation failed: "
                                         + std::string(cudaGetErrorString(err)));
            }
            return ptr;
        }

        void cuda_free(void* ptr) {
            if (ptr != nullptr) {
                cudaError_t err = cudaFree(ptr);
                if (err != cudaSuccess) {
                    throw std::runtime_error("CUDA device memory deallocation failed: "
                                             + std::string(cudaGetErrorString(err)));
                }
            }
        }

    };  // namespace internal
};      // namespace MT

#ifdef ENABLE_CUDA
#    include <cuda_runtime.h>
#else
#    include <cstring>
#endif

#include <cstdlib>
#include <iostream>
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
            if (::MT::cuda::is_available()) {
                cudaError_t err = cudaMalloc(&ptr, size);
                if (err != cudaSuccess) {
                    throw std::runtime_error("CUDA device memory allocation failed: "
                                             + std::string(cudaGetErrorString(err)));
                }
            } else
#endif
            {

                constexpr size_t CUDA_DEFAULT_ALIGNMENT = 256;
                ptr = std::aligned_alloc(CUDA_DEFAULT_ALIGNMENT, (size + CUDA_DEFAULT_ALIGNMENT - 1)
                                                                     / CUDA_DEFAULT_ALIGNMENT
                                                                     * CUDA_DEFAULT_ALIGNMENT);
            }
            if (ptr == nullptr) {
                throw std::bad_alloc();
            }
            return ptr;
        }

        void cuda_free(void* ptr) {
            if (ptr != nullptr) {
#ifdef ENABLE_CUDA
                if (::MT::cuda::is_available()) {
                    cudaError_t err = cudaFree(ptr);
                    if (err != cudaSuccess) {
                        throw std::runtime_error("CUDA device memory deallocation failed: "
                                                 + std::string(cudaGetErrorString(err)));
                    }
                } else
#endif
                {
                    std::free(ptr);
                }
            }
        }

    };  // namespace internal

    namespace cuda {
        namespace internal {
            bool _is_available() {
#ifdef ENABLE_CUDA
                int         device_count = 0;
                cudaError_t err          = cudaGetDeviceCount(&device_count);
                if (err != cudaSuccess) {
                    return false;
                }
                return device_count > 0;
#else
                return false;
#endif
            }
        };  // namespace internal
        bool is_available() {
#ifdef ENABLE_CUDA
            static const bool state = internal::_is_available();
            return state;
#else
            return false;
#endif
        }

        std::pair<size_t, size_t> get_device_memory_usage() {
#ifdef ENABLE_CUDA
            if (::MT::cuda::is_available()) {
                size_t      free_mem, total_mem;
                cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
                if (err != cudaSuccess) {
                    throw std::runtime_error("Failed to get CUDA device memory usage: "
                                             + std::string(cudaGetErrorString(err)));
                }
                return {free_mem, total_mem};
            }
#endif
            return {0, 0};
        }

        std::pair<int, int> device_get_stream_priority_range() {
#ifdef ENABLE_CUDA
            if (::MT::cuda::is_available()) {
                int         low_priority, high_priority;
                cudaError_t err = cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority);
                if (err != cudaSuccess) {
                    throw std::runtime_error("Failed to get CUDA stream priority range: "
                                             + std::string(cudaGetErrorString(err)));
                }
                return {low_priority, high_priority};
            }
#endif
            return {0, 0};
        }
        void set_device_flags(cudaDeviceFlag flags) {
#ifdef ENABLE_CUDA

            if (::MT::cuda::is_available()) {
                using _Flag = decltype(cudaDeviceScheduleAuto);
                static_assert(_Flag(cudaDeviceFlag::ScheduleAuto) == cudaDeviceScheduleAuto);
                static_assert(_Flag(cudaDeviceFlag::ScheduleSpin) == cudaDeviceScheduleSpin);
                static_assert(_Flag(cudaDeviceFlag::ScheduleYield) == cudaDeviceScheduleYield);
                static_assert(_Flag(cudaDeviceFlag::ScheduleBlockingSync)
                              == cudaDeviceScheduleBlockingSync);
                static_assert(_Flag(cudaDeviceFlag::ScheduleMask) == cudaDeviceScheduleMask);
                static_assert(_Flag(cudaDeviceFlag::MapHost) == cudaDeviceMapHost);
                static_assert(_Flag(cudaDeviceFlag::LmemResizeToMax) == cudaDeviceLmemResizeToMax);
                static_assert(_Flag(cudaDeviceFlag::SyncMemops) == cudaDeviceSyncMemops);
                static_assert(_Flag(cudaDeviceFlag::Mask) == cudaDeviceMask);
                cudaError_t err = cudaSetDeviceFlags(_Flag(flags));
                if (err != cudaSuccess) {
                    throw std::runtime_error("Failed to set CUDA device flags: "
                                             + std::string(cudaGetErrorString(err)));
                }
            }
#endif
        }

        void host_register(void* ptr, size_t bytes, cudaHostRegisterFlag flags) {
#ifdef ENABLE_CUDA
            if (::MT::cuda::is_available()) {
                using _cudaFlag = decltype(cudaHostRegisterDefault);

                static_assert(_cudaFlag(cudaHostRegisterFlag::Default) == cudaHostRegisterDefault);
                static_assert(_cudaFlag(cudaHostRegisterFlag::Portable)
                              == cudaHostRegisterPortable);
                static_assert(_cudaFlag(cudaHostRegisterFlag::Mapped) == cudaHostRegisterMapped);
                static_assert(_cudaFlag(cudaHostRegisterFlag::IoMemory)
                              == cudaHostRegisterIoMemory);
                static_assert(_cudaFlag(cudaHostRegisterFlag::ReadOnly)
                              == cudaHostRegisterReadOnly);
                cudaError_t err = cudaHostRegister(ptr, bytes, _cudaFlag(flags));
                if (err != cudaSuccess) {
                    throw std::runtime_error("Failed to register host memory: "
                                             + std::string(cudaGetErrorString(err)));
                }
            }
#endif
        }

        void* context::cuda_memset(void* dest, int ch, size_t n) const {
#ifdef ENABLE_CUDA
            if (::MT::cuda::is_available()) {
                cudaError_t err = cudaSuccess;
                if (async_) {
                    err = cudaMemsetAsync(dest, ch, n, static_cast<::cudaStream_t>(stream_));
                } else {
                    err = cudaMemset(dest, ch, n);
                }
                if (err != cudaSuccess) {
                    throw std::runtime_error("CUDA device memory memset failed: "
                                             + std::string(cudaGetErrorString(err)));
                }
            } else
#endif
            {
                std::memset(dest, ch, n);
            }

            return dest;
        }

        void* context::cuda_memcpy(void* dest, const void* src, size_t n,
                                   cudaMemcpyKind kind) const {
#ifdef ENABLE_CUDA
            if (::MT::cuda::is_available()) {
                cudaError_t err = cudaSuccess;

                static_assert(static_cast<::cudaMemcpyKind>(cudaMemcpyKind::cudaMemcpyHostToHost)
                              == cudaMemcpyHostToHost);
                static_assert(static_cast<::cudaMemcpyKind>(cudaMemcpyKind::cudaMemcpyHostToDevice)
                              == cudaMemcpyHostToDevice);
                static_assert(static_cast<::cudaMemcpyKind>(cudaMemcpyKind::cudaMemcpyDeviceToHost)
                              == cudaMemcpyDeviceToHost);
                static_assert(
                    static_cast<::cudaMemcpyKind>(cudaMemcpyKind::cudaMemcpyDeviceToDevice)
                    == cudaMemcpyDeviceToDevice);

                const auto _kind = static_cast<::cudaMemcpyKind>(kind);
                if (async_) {
                    err = cudaMemcpyAsync(dest, src, n, _kind,
                                          static_cast<::cudaStream_t>(stream_));
                } else {
                    err = cudaMemcpy(dest, src, n, _kind);
                }

                if (err != cudaSuccess) {
                    throw std::runtime_error("CUDA device memory memset failed: "
                                             + std::string(cudaGetErrorString(err)));
                }
            } else
#endif
            {
                std::memcpy(dest, src, n);
            }

            return dest;
        }

        std::unique_ptr<context> context::new_with_priority(int priority, bool async) {
#ifdef ENABLE_CUDA
            if (::MT::cuda::is_available()) {
                ::cudaStream_t _stream;
                cudaError_t    err
                    = cudaStreamCreateWithPriority(&_stream, cudaStreamDefault, priority);
                if (err != cudaSuccess) {
                    throw std::runtime_error("Failed to create CUDA stream with priority: "
                                             + std::string(cudaGetErrorString(err)));
                }
                auto c     = std::make_unique<context>(async);
                c->stream_ = _stream;

                return std::move(c);
            }
#endif
            return std::make_unique<context>();
        }


        void context::release() {
#ifdef ENABLE_CUDA
            if (stream_ != nullptr) {
                if (::MT::cuda::is_available()) {
                    cudaError_t err = cudaStreamDestroy(static_cast<::cudaStream_t>(stream_));
                    stream_         = nullptr;  // avoid double throw even if it fail to release
                    if (err != cudaSuccess) {
                        throw std::runtime_error("Failed to release the related stream: "
                                                 + std::string(cudaGetErrorString(err)));
                    }
                }
            }
#endif
        }

        context::~context() {
            try {
                release();
            } catch (const std::exception& e) {
                std::cerr << "Exception in context destructor: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "Unknown exception in context destructor." << std::endl;
            }
        }

        void context::synchronize() {
#ifdef ENABLE_CUDA
            if (::MT::cuda::is_available()) {
                cudaError_t err = cudaStreamSynchronize(static_cast<::cudaStream_t>(stream_));
                if (err != cudaSuccess) {
                    throw std::runtime_error("Failed to synchronize CUDA stream: "
                                             + std::string(cudaGetErrorString(err)));
                }
            }
#endif
        }

    };  // namespace cuda

};  // namespace MT

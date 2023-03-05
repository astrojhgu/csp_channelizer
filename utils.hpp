#ifndef UTILS_HPP_
#define UTILS_HPP_
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <utility>
#include "types.hpp"
extern size_t gpu_mem_used;

static constexpr FloatType PI = 3.14159265358979323846f;

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CUFFT_CALL(call)                                                                                               \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>(call);                                                                  \
        if (status != CUFFT_SUCCESS)                                                                                   \
            fprintf(stderr,                                                                                            \
                    "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                           \
                    "with "                                                                                            \
                    "code (%d).\n",                                                                                    \
                    #call,                                                                                             \
                    __LINE__,                                                                                          \
                    __FILE__,                                                                                          \
                    status);                                                                                           \
    }

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char *const file, const int line);

#define CHECK_CUDA(error)                                                                                              \
    {                                                                                                                  \
        if (error != cudaSuccess) {                                                                                    \
            fprintf(stderr, "Error: %s:%d\n", __FILE__, __LINE__);                                                     \
            fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));                               \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    }

/**
 * @brief check pointer validity
 * @param ptr: generic pointer
 */
#define CHECK_PTR(ptr)                                                                                                 \
    {                                                                                                                  \
        if (ptr == NULL) {                                                                                             \
            fprintf(stderr, "Error: %s:%d\n", __FILE__, __LINE__);                                                     \
            fprintf(stderr, "Null pointer\n");                                                                         \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    }

template <typename T>
class Deleter {
  public:
    void operator()(T *ptr) {
        cudaFree((void *) ptr);
        std::cout << "\nDeleted\n";
    }
};

template <typename T>
std::unique_ptr<T[], Deleter<T>> gpu_alloc(std::size_t n) {
    T *ptr = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&ptr, sizeof(T) * n));
    gpu_mem_used += sizeof(T) * n;

    // auto deleter = [&](auto *ptr) { cudaFree((void *)ptr); };
    Deleter<T> deleter;
    return std::unique_ptr<T[], Deleter<T>>(ptr, deleter);
}

template <typename T, typename U>
void cuda_mem_cpy(T *dst, const U *src, size_t nelem, cudaMemcpyKind dir) {
    assert(sizeof(T) == sizeof(U));
    CHECK_CUDA_ERROR(cudaMemcpy(dst, src, nelem * sizeof(T), dir));
}

#endif

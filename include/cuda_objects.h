#ifndef CUDA_OBJECTS_H
#define CUDA_OBJECTS_H


#include <cuda_runtime.h>
#include <memory>

namespace cuda {

    class CudaException : public std::exception
    {
    private:
        cudaError_t m_error;

    public:
        CudaException(cudaError_t err) : m_error(err) { }

        virtual const char* what() const override
        {
            return cudaGetErrorString(m_error);
        }

        static void ThrowOnError(cudaError_t err)
        {
            if (err != cudaSuccess) {
                throw CudaException(err);
            }
        }
    };

    class CudaStream
    {
    private:
        cudaStream_t m_handle;

    public:
        explicit CudaStream(unsigned int flags = cudaStreamDefault) : m_handle(nullptr)
        {
            CudaException::ThrowOnError(
                cudaStreamCreateWithFlags(&m_handle, flags)
                );
        }

        cudaStream_t GetHandle() const { return m_handle; }

        ~CudaStream()
        {
            if (m_handle != nullptr) {
                CudaException::ThrowOnError(
                    cudaStreamDestroy(m_handle)
                    );
            }
        }
    };

    struct CudaArrayBase
    {
        virtual void* PtrToKernelArg() = 0;
        virtual void Invalidate(cudaStream_t stream = (cudaStream_t)0) = 0;
        virtual void Gather(cudaStream_t stream = (cudaStream_t)0) = 0;

        virtual ~CudaArrayBase() { }
    };

    template <typename T>
    class CudaArray : public CudaArrayBase
    {
    private:
        T* m_dev_ptr;
        T* m_host_ptr;

        const int m_size;
        const bool m_read_only; // Host buffer is read only

    public:
        CudaArray(T* ptr, int size)
            : m_dev_ptr(nullptr), m_host_ptr(ptr), m_size(size), m_read_only(false)
        {
            CudaException::ThrowOnError(
                cudaMalloc(&m_dev_ptr, m_size * sizeof(T))
                );
        }
        CudaArray(const T* ptr, int size)
            : m_dev_ptr(nullptr), m_host_ptr((T*)ptr), m_size(size), m_read_only(true)
        {
            CudaException::ThrowOnError(
                cudaMalloc(&m_dev_ptr, m_size * sizeof(T))
                );
        }

        void* PtrToKernelArg() override { return (void*)&m_dev_ptr; }

        void Invalidate(cudaStream_t stream = (cudaStream_t)0) override
        {
            CudaException::ThrowOnError(
                cudaMemcpyAsync(m_dev_ptr, m_host_ptr, m_size * sizeof(T), cudaMemcpyHostToDevice, stream)
                );
        }

        void Gather(cudaStream_t stream = (cudaStream_t)0) override
        {
            if (m_read_only) {
                throw std::exception("Host buffer is read only, cannot Gather.");
            }

            CudaException::ThrowOnError(
                cudaMemcpyAsync(m_host_ptr, m_dev_ptr, m_size * sizeof(T), cudaMemcpyDeviceToHost, stream)
                );
        }

        virtual ~CudaArray()
        {
            CudaException::ThrowOnError(
                cudaFree(m_dev_ptr)
                );
        }
    };

    template<typename T>
    struct InputArray
    {
        const T* m_ptr;
        int m_size;

        InputArray(const T* ptr, int size) : m_ptr(ptr), m_size(size) { }
        InputArray(const std::vector<T>& vec) : m_ptr(&vec[0]), m_size(vec.size()) { }
    };

    template<typename T>
    struct OutputArray
    {
        T* m_ptr;
        int m_size;

        OutputArray(T* ptr, int size) : m_ptr(ptr), m_size(size) { }
        OutputArray(const std::vector<T>& vec) : m_ptr(&vec[0]), m_size(vec.size()) { }
    };
}

#endif // CUDA_OBJECTS_H
#ifndef CUDA_CONTEXT_H
#define CUDA_CONTEXT_H


#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <stack>
#include <functional>
#include <chrono>

#include "stopwatch.h"
#include "cuda_objects.h"

namespace cuda {

    class CudaContext
    {
    private:
        enum KernelArgType { Input, Output /*, Symbol*/ };
        struct KernelArg
        {
            std::unique_ptr<CudaArrayBase> arr;
            KernelArgType type;
            std::function<int()> regression;

            KernelArg(CudaArrayBase* arr, KernelArgType type, decltype(regression) regres)
                : arr(arr), type(type), regression(regres) { }

            KernelArg(const KernelArg& other) = delete;
        };

        std::vector< std::unique_ptr<KernelArg> > m_args;

        void* m_kernel;
        dim3 m_grid_dim;
        dim3 m_block_dim;

        std::stack<cudaStream_t> m_streams;
        std::stack<Stopwatch*> m_stopwatches;

    public:
        CudaContext()
        {
            m_streams.push((cudaStream_t)0);
        }

        template <typename T>
        CudaContext& RegisterInput(const T* ptr, int size)
        {
            CudaArray<T>* arr = new CudaArray<T>(ptr, size);
            m_args.emplace_back(new KernelArg(arr, Input, [] { return 0; }));

            return *this;
        }

        template <typename T>
        CudaContext& RegisterInput(const std::vector<T>& vec)
        {
            return RegisterInput(&vec[0], vec.size());
        }

        template <typename T>
        CudaContext& RegisterOutput(
            T* ptr, int size,
            std::function<int(const T*, int)> regression = [](const T*, int) { return 0; }
            )
        {
            CudaArray<T>* arr = new CudaArray<T>(ptr, size);
            m_args.emplace_back(new KernelArg(arr, Output, std::bind(regression, ptr, size)));

            return *this;
        }

        template <typename T>
        CudaContext& RegisterOutput(
            const std::vector<T>& vec,
            std::function<int(const T*, int)> regression = [](const T*, int) { return 0; }
            )
        {
            return RegisterOutput(&vec[0], vec.size(), regression);
        }

        CudaContext& InvalidateInputs()
        {
            for (auto&& arg : m_args)
            {
                if (arg->type == Input) {
                    arg->arr->Invalidate(m_streams.top());
                }
            }

            return *this;
        }

        CudaContext& PushStream(cudaStream_t stream)
        {
            m_streams.push(stream);
            return *this;
        }

        CudaContext& PushStream(const CudaStream& stream)
        {
            m_streams.push(stream.GetHandle());
            return *this;
        }

        CudaContext& PushTiming(Stopwatch& stopwatch)
        {
            stopwatch.start();
            m_stopwatches.push(&stopwatch);
            return *this;
        }

        template <typename Kernel>
        CudaContext& RegisterKernel(Kernel kernel, dim3 gridDim, dim3 blockDim)
        {
            m_kernel = (void*)kernel;
            m_grid_dim = gridDim;
            m_block_dim = blockDim;

            return *this;
        }

        template<typename T>
        inline void RegisterArgs(const InputArray<T>& input)
        {
            RegisterInput(input.m_ptr, input.m_size);
        }

        template<typename T>
        inline void RegisterArgs(const OutputArray<T>& output)
        {
            RegisterOutput(output.m_ptr, output.m_size);
        }

        inline void RegisterArgs()
        {
        }

        template <typename First, typename... Args>
        inline void RegisterArgs(const First& first, const Args&... args)
        {
            RegisterArgs(first);
            RegisterArgs(args...);
        }

        template <typename Kernel, typename... Args>
        CudaContext& RegisterKernel(Kernel kernel, dim3 gridDim, dim3 blockDim, const Args&... args)
        {
            m_kernel = (void*)kernel;
            m_grid_dim = gridDim;
            m_block_dim = blockDim;

            RegisterArgs(args...);

            return *this;
        }

        CudaContext& Launch(int iterations = 1)
        {
            std::vector<void*> ptrs(m_args.size());
            int i = 0;

            for (auto&& arg : m_args) {
                ptrs[i++] = arg->arr->PtrToKernelArg();
            }

            for (int j = 0; j < iterations; j++)
            {
                CudaException::ThrowOnError(
                    cudaLaunchKernel(m_kernel, m_grid_dim, m_block_dim, &ptrs[0], 0, m_streams.top())
                    );
            }

            return *this;
        }

        CudaContext& Sync()
        {
            CudaException::ThrowOnError(
                cudaStreamSynchronize(m_streams.top())
                );

            return *this;
        }

        CudaContext& PopStream()
        {
            if (m_streams.size() > 1) {
                m_streams.pop();
            }

            return *this;
        }

        CudaContext& PopTiming()
        {
            m_stopwatches.top()->stop();
            m_stopwatches.pop();

            return *this;
        }

        CudaContext& GatherOutputs()
        {
            for (auto&& arg : m_args)
            {
                if (arg->type == Output) {
                    arg->arr->Gather(m_streams.top());
                }
            }

            return *this;
        }

        CudaContext& Verify()
        {
            bool succedded = true;
            for (auto&& arg : m_args)
            {
                if (arg->type == Output) {
                    int errors = arg->regression();
                    succedded &= (errors == 0);
                }
            }

            if (!succedded) {
                // TODO: Add more info
                throw std::exception("Verification failed");
            }

            return *this;
        }

        void Cleanup()
        {
            m_args.clear();

            while (m_streams.size() > 1) {
                m_streams.pop();
            }
        }
    };
}

#endif // CUDA_CONTEXT_H
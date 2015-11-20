
#include "../include/cuda_context.h"

__global__ void AddKernel(const int *a, const int *b, int *c)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void Scratchpad()
{
    const int size = 5;
    const int a[size] = { 1, 2, 3, 4, 5 };
    const int b[size] = { 10, 20, 30, 40, 50 };
    int c[size] = { 0 };

    Stopwatch sw;
    cuda::CudaStream stream(cudaStreamNonBlocking);

    std::function<int(const int*, int)> regression = [size, &a, &b](const int* d, int)
    {
        int errs = 0;
        for (int i = 0; i < size; i++) {
            if (d[i] != a[i] + b[i])
                ++errs;
        }

        return errs;
    };

    cuda::CudaContext()
        .RegisterKernel(AddKernel, dim3(1), dim3(size), cuda::InputArray<int>(a, size), cuda::InputArray<int>(b, size), cuda::OutputArray<int>(c, size))
        .PushStream(stream)
        .InvalidateInputs()
        .PushTiming(sw)
        .Launch(10000)
        .Sync()
        .PopTiming()
        .GatherOutputs()
        .Sync()
        .Verify()
        .PopStream();

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    printf("\nOperation timing: %f ms\n", sw.ms());
}

int main()
{
    try {
        Scratchpad();
    }

    catch (const std::exception& ex)  {
        printf("%s\n", ex.what());
    }

    return 0;
}
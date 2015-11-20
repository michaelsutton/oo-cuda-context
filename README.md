Run CUDA kernels with a few lines of code

Usage example
-------

    cuda::CudaContext()
        .RegisterKernel(AddKernel, dim3(1), dim3(size), cuda::Input(a, size), cuda::Input(b, size), cuda::Output(c, size))
        .InvalidateInputs()
        .Launch()
        .GatherOutputs()
        .Sync();

Requirements
------------

C++ 11.
CUDA 7.0 or higher.

This is just a minor kikoff for a possible complete CUDA wrapper. Ideas and contributions would be welcomed.




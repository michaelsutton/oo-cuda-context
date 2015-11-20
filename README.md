Run CUDA kernels with a few lines of code

This is just a minor kikoff for a possible complete CUDA wrapper. Further ideas and contributions would be welcomed.

Usage example
-------
```c++
__global__ void AddKernel(const int *a, const int *b, int *c) { ... }
    
int size = ...
int *a = ... , *b = ..., *c = ... // Allocated host buffers 

cuda::CudaContext()
    .RegisterKernel(AddKernel, dim3(1), dim3(size), cuda::Input(a, size), cuda::Input(b, size), cuda::Output(c, size))
    .InvalidateInputs()
    .Launch()
    .GatherOutputs()
    .Sync();
```
Requirements
------------

C++ 11.
CUDA 7.0 or higher.





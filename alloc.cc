// #include <sys/types.h>
// #include <cuda_runtime_api.h>
// #include <iostream>
// // Compile with g++ alloc.cc -o alloc.so -I/usr/local/cuda/include -shared -fPIC
// extern "C" {
// void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
//    void *ptr;
//    cudaMalloc(&ptr, size);
//    std::cout<<"alloc "<<ptr<<size<<std::endl;
//    return ptr;
// }

// void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
//    std::cout<<"free "<<ptr<< " "<<stream<<std::endl;
//    cudaFree(ptr);
// }
// }
#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <iostream>

// Compile with: g++ alloc.cc -o alloc.so -I/usr/local/cuda/include -shared -fPIC -lcudart
extern "C" {
    void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
        void *ptr;
        cudaMalloc(&ptr, size);
      //   cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
      //   std::cout << "alloc " << ptr << " " << size << std::endl;
        return ptr;
    }

    void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
      //   std::cout << "free " << ptr << " " << size << " " << stream << std::endl;
        cudaFree(ptr);
    }
}
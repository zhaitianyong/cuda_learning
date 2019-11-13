
#include <stdio.h>
// CUDA runtime
#include <cuda_runtime.h>

// __global__ 声明 gpu 线程调用

__global__ void sum(int *a, int *b, int *c ){
    
    c[0] = a[0] + b[0];
}


int main(int argc, char **argv)
{

    
    // 声明 Host 变量
    int a[1]={1},b[1] ={2},c[1]={0};
    
    // 声明 device 变量
    int *gpu_a, *gpu_b, *gpu_c;
    
    // 开辟空间
    cudaMalloc((void **)&gpu_a, sizeof(int));
    cudaMalloc((void **)&gpu_b, sizeof(int));
    cudaMalloc((void **)&gpu_c, sizeof(int));
    
    // 讲Host 数据上载到gpu上
    cudaMemcpy(gpu_a, a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, sizeof(int), cudaMemcpyHostToDevice);
    
    // 执行
    sum<<<1, 1>>>(gpu_a, gpu_b, gpu_c);
    
    // 将执行结果 下载到Host 变量 c中
    cudaMemcpy(c, gpu_c, sizeof(int), cudaMemcpyDeviceToHost);
    
    // 释放空间
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    
    // 打印
    printf(" %d + %d = %d \n", a[0], b[0], c[0]);
    

    return 0;
}

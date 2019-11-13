
#include <stdio.h>

#include <cuda_runtime.h>


__global__ void add(const int *a, const int *b, int *c, int size){
 
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < size){
     c[i] = a[i] + b[i];  
    }
}

int main(void)
{
        int devID=0;
    cudaDeviceProp props;

    // This will pick the best possible CUDA capable device
    // devID = findCudaDevice(argc, (const char **)argv);

    //Get GPU information
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&props, devID);
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           devID, props.name, props.major, props.minor);

    printf("printf() is called. Output:\n\n");
    
    int num = 50000;
    unsigned long size = num * sizeof(int);
    int A[num], B[num], C[num];
    
    
    for (int i=0; i < num; ++i)
    {
      A[i] =  i;
      B[i] =  i;
    }
    
    int *gpuA, *gpuB, *gpuC;
    
    cudaMalloc((void **)&gpuA, size);
    cudaMalloc((void **)&gpuB, size);
    cudaMalloc((void **)&gpuC, size);
    
    
    cudaMemcpy(gpuA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, B, size, cudaMemcpyHostToDevice);
    
    
    int threadsPerBlock = 256; // 每个block拥有的线层数量
    int blocksPerGird = (num + threadsPerBlock -1)/threadsPerBlock;  // 多少个block
    
    add<<<blocksPerGird, threadsPerBlock>>>(gpuA, gpuB, gpuC, num);
    
    
    cudaMemcpy(C, gpuC, size, cudaMemcpyDeviceToHost);
    
    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);


    for(int i=0; i < 10; ++i){
      printf("%d ", C[i]); 
    }
    
    return 0;
}

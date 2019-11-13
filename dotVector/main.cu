#include <stdio.h>
#include <cuda_runtime.h>

#define MIN(a, b) (a > b? b: a)

const int N = 32 * 1024;
const int threadsPerBlock = 256; // 每个block下的线程数
const int blocksPerGrid = MIN(32, (N + threadsPerBlock -1)/threadsPerBlock); // 分配的block数

// 点乘
__global__ void dot(float *a, float *b, float *c){
    __shared__ float cache[threadsPerBlock];  // 共享内存 每个block下的共享内存
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 当前线程的编号
    
    int cacheIndex = threadIdx.x;
    
    float temp =0;
    
    while(tid < N){
        temp += a[tid] * b[tid];
        tid += gridDim.x * blockDim.x; 
    }
    
    cache[cacheIndex] = temp;
    
    __syncthreads();
    
    // 规约每个线程块内的点积和
    
    int i = blockDim.x /2;
    
    while(i != 0){
        if ( cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    
    }
    
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
    

}


int main(void){
    float *a, *b, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    
    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    partial_c = (float *)malloc(blocksPerGrid*sizeof(float));

    // 赋值
    for(int i=0;i< N; ++i){
        a[i] = (float)i;
        b[i] = i * 2.0;
    }
    // device 上分配内存
    cudaMalloc((void **)&dev_a, N * sizeof(float));
    cudaMalloc((void **)&dev_b, N * sizeof(float));
    cudaMalloc((void **)&dev_partial_c, blocksPerGrid * sizeof(float));

    // host 复制到 device上
    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 执行
    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
    
    
    // device 数据下载到Host 
    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    
    //求和
    
    float sum=0.0;
    for(int i =0; i<blocksPerGrid; ++i)
    {
        sum += partial_c[i];
    }
    
    printf("a dot b is %.2g \n", sum);
    
    //释放内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);
    
    free(a);
    free(b);
    free(partial_c);
    

 return 0;
}

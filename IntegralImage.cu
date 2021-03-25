#include "integralimage.h"

IntegralCuda::IntegralCuda(){

}

void IntegralCuda::h_mem_init()
{

    h_origin_image = (int*)malloc(size_int);
    h_result_phase1 = (int*)malloc(size_int);
    h_result_phase2  = (int*)malloc(size_int);
    h_result_phase3 = (int*)malloc(size_int);



    // initial h_weight, h_flow from input

}

void IntegralCuda::d_mem_init()
{
    gpuErrChk(cudaMalloc((void**)&d_origin_image, size_int));
    gpuErrChk(cudaMalloc((void**)&d_result_phase1, size_int));
    gpuErrChk(cudaMalloc((void**)&d_result_phase2, size_int));
    gpuErrChk(cudaMalloc((void**)&d_result_phase3, size_int));
}

void IntegralCuda::cudaMemoryInit(int cols, int rows){
    width = cols;
    height = rows;
    image_size = width*height;
    size_int = image_size*sizeof(int);
    h_mem_init();
    d_mem_init();
}

void IntegralCuda::cudaCutsFreeMem(){
    free(h_origin_image);
    free(h_result_phase1);
    free(h_result_phase2);
    free(h_result_phase3);

    gpuErrChk(cudaFree(d_origin_image));
    gpuErrChk(cudaFree(d_result_phase1));
    gpuErrChk(cudaFree(d_result_phase2));
    gpuErrChk(cudaFree(d_result_phase3));
}

__global__ void
integral_phase1_kernel1(int *d_origin_image, int * d_result_phase1, int *d_horizon, int width, int N){
    __shared__ int XY[33*16];
    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int tid = iy*width + ix;
    //if(x1 >= 20 || y1 >= 20) return;

    XY[y1*33 + x1] = d_origin_image[tid];
    __syncthreads();
    int a, b;
    for(int stride = 1; stride < blockDim.x; stride *= 2){
        a = x1 + stride;
        b = 2*stride;
        if(a%b < stride){
            XY[y1*33 + x1] += XY[y1*33 + (a/b)*b - stride - 1];
        }
        __syncthreads();
    }

    for(int stride = 1; stride < blockDim.y; stride *= 2){
        a = y1 + stride;
        b = 2*stride;
        if(a%b < stride){
            XY[y1*33 + x1] += XY[((a/b)*b - stride - 1)*33 + x1];
        }
        __syncthreads();
    }

    d_result_phase1[tid] = XY[y1*33+x1];
    if(x1 == blockDim.x-1)
        d_horizon[y1*gridDim.x + blockIdx.x] = XY[y1*33+x1];
}

__global__ void
horizon_sum_kernel(int *d_horizon_src, int *d_horizon_dst, int width, int N){
    __shared__ int XY[33*16];
    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int tid = iy*width + ix;
    if(x1 >= 30) return;

    XY[y1*33 + x1] = d_horizon_src[tid];
    __syncthreads();
    int a, b;
    for(int stride = 1; stride < blockDim.x; stride *= 2){
        a = x1 + stride;
        b = 2*stride;
        if(a%b < stride){
            XY[y1*33 + x1] += XY[y1*33 + (a/b)*b - stride - 1];
        }
        __syncthreads();
    }

    d_horizon_dst[tid] = XY[y1*33+x1];
}

__global__ void
integral_phase2_kernel(int *d_result_phase1, int * d_result_phase2, int *d_horizon_dst, int width, int N){
    __shared__ int XY[33*16];
    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int tid = iy*width + ix;
    //if(x1 >= 20 || y1 >= 20) return;

    XY[y1*33 + x1] = d_result_phase1[tid];
    __syncthreads();
    int a, b;
    for(int stride = 1; stride < blockDim.x; stride *= 2){
        a = x1 + stride;
        b = 2*stride;
        if(a%b < stride){
            XY[y1*33 + x1] += XY[y1*33 + (a/b)*b - stride - 1];
        }
        __syncthreads();
    }

    for(int stride = 1; stride < blockDim.y; stride *= 2){
        a = y1 + stride;
        b = 2*stride;
        if(a%b < stride){
            XY[y1*33 + x1] += XY[((a/b)*b - stride - 1)*33 + x1];
        }
        __syncthreads();
    }

    d_result_phase2[tid] = XY[y1*33+x1];
    if(x1 == blockDim.x-1)
        d_horizon[y1*gridDim.x + blockIdx.x] = XY[y1*33+x1];
}

void IntegralCuda::prefixSum2D(){
//    cv::Mat m1;
//    src.convertTo(m1, CV_32SC1);
//    memcpy(h_origin_image, m1.ptr(0), size_int);
    for(int i=0; i<image_size; i++){
        h_origin_image[i] = 1;
    }
    writeToFile("../variable/h_origin_image.txt", h_origin_image, width, height);
    gpuErrChk(cudaMemcpy(d_origin_image, h_origin_image, size_int, cudaMemcpyHostToDevice));

    dim3 block(32,16,1);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y, 1);
    integral_kernel1<<<grid, block>>>(d_origin_image, d_result_phase1, width, image_size);

    gpuErrChk(cudaMemcpy(h_result_phase1, d_result_phase1, size_int, cudaMemcpyDeviceToHost));
    writeToFile("../variable/h_result_phase1.txt", h_result_phase1, width, height);

}

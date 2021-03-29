#include "integralimage.h"

IntegralCuda::IntegralCuda(){

}

void IntegralCuda::h_mem_init()
{

    h_origin_image = (int*)malloc(size_int);
    h_result_phase1 = (int*)malloc(size_int);
    h_result_phase2  = (int*)malloc(size_int);
    h_result_phase3 = (int*)malloc(size_int);
    h_horizon_src = (int*)malloc(horizon_size*sizeof(int));
    h_horizon_dst = (int*)malloc(horizon_size*sizeof(int));
    h_vertical_src = (int*)malloc(vertical_size*sizeof(int));
    h_vertical_dst = (int*)malloc(vertical_size*sizeof(int));


    // initial h_weight, h_flow from input

}

void IntegralCuda::d_mem_init()
{
    gpuErrChk(cudaMalloc((void**)&d_origin_image, size_int));
    gpuErrChk(cudaMalloc((void**)&d_result_phase1, size_int));
    gpuErrChk(cudaMalloc((void**)&d_result_phase2, size_int));
    gpuErrChk(cudaMalloc((void**)&d_result_phase3, size_int));
    gpuErrChk(cudaMalloc((void**)&d_horizon_src, horizon_size*sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&d_horizon_dst, horizon_size*sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&d_vertical_src, vertical_size*sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&d_vertical_dst, vertical_size*sizeof(int)));
}

void IntegralCuda::cudaMemoryInit(int cols, int rows){
    width = cols;
    height = rows;
    image_size = width*height;
    size_int = image_size*sizeof(int);
    horizon_width = width/BLOCKDIMX; // 20
    horizon_size = horizon_width*height; // 20x480
    vertical_width = height/BLOCKDIMY; // 15
    vertical_size = vertical_width*width; // 640x15
    h_mem_init();
    d_mem_init();
}

void IntegralCuda::cudaCutsFreeMem(){
    free(h_origin_image);
    free(h_result_phase1);
    free(h_result_phase2);
    free(h_result_phase3);
    free(h_horizon_src);
    free(h_horizon_dst);
    free(h_vertical_src);
    free(h_vertical_dst);

    gpuErrChk(cudaFree(d_origin_image));
    gpuErrChk(cudaFree(d_result_phase1));
    gpuErrChk(cudaFree(d_result_phase2));
    gpuErrChk(cudaFree(d_result_phase3));
    gpuErrChk(cudaFree(d_horizon_src));
    gpuErrChk(cudaFree(d_horizon_dst));
    gpuErrChk(cudaFree(d_vertical_src));
    gpuErrChk(cudaFree(d_vertical_dst));
}

// prefix sum per block
__global__ void
integral_phase1_kernel(int *d_origin_image, int * d_result_phase1, int *d_horizon_src, int width, int N){
    __shared__ int XY[33*16];
    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int tid = iy*width + ix;
    int a1 = y1*33, index;
    //if(x1 >= 20 || y1 >= 20) return;

    XY[a1 + x1] = d_origin_image[tid];
    __syncthreads();
    int a, b;
    for(int stride = 1; stride < blockDim.x; stride *= 2){
        a = x1 + stride;
        b = 2*stride;
        if(a%b < stride){
            XY[a1 + x1] += XY[a1 + (a/b)*b - stride - 1];
        }
        __syncthreads();
    }

    for(int stride = 1; stride < blockDim.y; stride *= 2){
        a = y1 + stride;
        b = 2*stride;
        if(a%b < stride){
            XY[a1 + x1] += XY[((a/b)*b - stride - 1)*33 + x1];
        }
        __syncthreads();
    }
    // unrolling horizon
//    if(blockDim.x >= 32 && x1 < 16){
//        index = (x1+1)*2 - 1;
//        XY[a+index] += XY[a+index - 1];
//    }
//    __syncthreads();

//    if(blockDim.x >= 16 && x1 < 8){
//        index = (x1+1)*4 - 1;
//        XY[a+index] += XY[a+index - 2];
//    }
//    __syncthreads();
//    if(blockDim.x >= 8 && x1 < 4){
//        index = (x1+1)*8 - 1;
//        XY[a+index] += XY[a+index - 4];
//    }
//    __syncthreads();
//    if(blockDim.x >= 4 && x1 < 2){
//        index = (x1+1)*16 - 1;
//        XY[a+index] += XY[a+index - 8];
//    }
//    __syncthreads();
//    if(blockDim.x >= 2 && x1 < 1){
//        index = (x1+1)*32 - 1;
//        XY[a+index] += XY[a+index - 16];
//    }
//    __syncthreads();
////    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
////        //__syncthreads();
////        int index = (threadIdx.x+1) * 2* stride -1;
////        if (index < blockDim.x) {
////            XY[y1*33 + index] += XY[y1*33 + index - stride];
////        }
////        __syncthreads();
////    }
//    for (int stride = blockDim.x/4; stride > 0; stride /= 2) {
//        //__syncthreads();
//        index = (threadIdx.x+1)*stride*2 - 1;
//        if(index + stride < blockDim.x) {
//            XY[y1*33 + index + stride] += XY[y1*33 + index];
//        }
//        __syncthreads();
//    }

//    //__syncthreads();
////    if(blockDim.y >= 32 && y1 < 16){
////        index = (y1+1)*2 - 1;
////        XY[index*33 + x1] += XY[(index-1)*33 + x1];
////    }
////    __syncthreads();

////    if(blockDim.y >= 16 && y1 < 8){
////        index = (y1+1)*4 - 1;
////        XY[index*33 + x1] += XY[(index-2)*33 + x1];
////    }
////    __syncthreads();
////    if(blockDim.y >= 8 && y1 < 4){
////        index = (y1+1)*8 - 1;
////        XY[index*33 + x1] += XY[(index-4)*33 + x1];
////    }
////    __syncthreads();
////    if(blockDim.y >= 4 && y1 < 2){
////        index = (y1+1)*16 - 1;
////        XY[index*33 + x1] += XY[(index-8)*33 + x1];
////    }
////    __syncthreads();
////    if(blockDim.y >= 2 && y1 < 1){
////        index = (y1+1)*32 - 1;
////        XY[index*33 + x1] += XY[(index-16)*33 + x1];
////    }
////    __syncthreads();
//    for (unsigned int stride = 1; stride < blockDim.y; stride *= 2) {
//        //__syncthreads();
//        index = (threadIdx.y+1) * 2* stride -1;
//        if (index < blockDim.y) {
//            XY[index*33 + x1] += XY[(index-stride)*33 + x1];
//        }
//        __syncthreads();
//    }
//    for (int stride = blockDim.y/4; stride > 0; stride /= 2) {
//        //__syncthreads();
//        int index = (threadIdx.y+1)*stride*2 - 1;
//        if(index + stride < blockDim.y) {
//            XY[(index+stride)*33 + x1] += XY[index*33 + x1];
//        }
//        __syncthreads();
//    }
    //__syncthreads();

    d_result_phase1[tid] = XY[a1+x1];
    if(x1 == blockDim.x-1)
        d_horizon_src[iy*gridDim.x + blockIdx.x] = XY[a1+x1];
}

__global__ void
horizon_sum_kernel(int *d_horizon_src, int *d_horizon_dst, int width, int horizon_width){
    __shared__ int XY[33*16];
    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int tid = iy*width + ix;

    if(ix >= horizon_width) return;
    int idx = iy*horizon_width + ix;

    XY[y1*33 + x1] = d_horizon_src[idx];
    __syncthreads();
    int a, b;
    for(int stride = 1; stride < horizon_width; stride *= 2){
        a = x1 + stride;
        b = 2*stride;
        if(a%b < stride){
            XY[y1*33 + x1] += XY[y1*33 + (a/b)*b - stride - 1];
        }
        __syncthreads();
    }

    d_horizon_dst[idx] = XY[y1*33+x1];
}

// prefix sum for horizon direction
__global__ void
integral_phase2_kernel(int *d_result_phase1, int * d_result_phase2, int *d_horizon_dst, int *d_vertical_src,
                       int width, int horizon_width, int vertical_width){
    __shared__ int XY[33*32];
    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int tid = iy*width + ix;
    XY[y1*33 + x1] = d_result_phase1[tid];
    __syncthreads();

    //if(x1 >= 20 || y1 >= 20) return;
    if(blockIdx.x != 0){
        __shared__ int XY1[2*32];
        if(x1 == 0)
            XY1[y1*2] = d_horizon_dst[iy*horizon_width + blockIdx.x-1];
        __syncthreads();

        XY[y1*33 + x1] += XY1[y1*2];
    }

    d_result_phase2[tid] = XY[y1*33+x1];
    if(y1 == blockDim.y-1)
        d_vertical_src[blockIdx.y*width + ix] = XY[y1*33+x1];
}

__global__ void
vertical_sum_kernel(int *d_vertical_src, int *d_vertical_dst, int width, int vertical_width){
    // hard code
    __shared__ int XY[33*15];
    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int tid = iy*width + ix;

    if(iy >= vertical_width) return;
    int idx = iy*width + ix;

    XY[y1*33 + x1] = d_vertical_src[idx];
    __syncthreads();
    int a, b;

    for(int stride = 1; stride < vertical_width; stride *= 2){
        a = y1 + stride;
        b = 2*stride;
        if(a%b < stride){
            XY[y1*33 + x1] += XY[((a/b)*b - stride - 1)*33 + x1];
        }
        __syncthreads();
    }
    d_vertical_dst[idx] = XY[y1*33+x1];
}

// prefix sum for vertical direction
__global__ void
integral_phase3_kernel(int *d_result_phase2, int * d_result_phase3, int *d_vertical_dst,
                       int width, int vertical_width){
    //__shared__ int XY[33*32];
    int x1 = threadIdx.x;
    int y1 = threadIdx.y;
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int tid = iy*width + ix;

    __syncthreads();
    if(blockIdx.y == 0)
        d_result_phase3[tid] = d_result_phase2[tid];

    //if(x1 >= 20 || y1 >= 20) return;
    if(blockIdx.y != 0){
        __shared__ int XY1[32*2];
        if(y1 == 0)
            XY1[x1] = d_vertical_dst[(blockIdx.y-1)*width + ix];
        __syncthreads();

        d_result_phase3[tid] = d_result_phase2[tid] +  XY1[x1];
    }

}

void IntegralCuda::prefixSum2D(int *d_origin_image1){

    //auto start = getMoment;
    for(int i=0; i<image_size; i++){
        h_origin_image[i] = 1;
    }
    //writeToFile("../variable/h_origin_image.txt", h_origin_image, width, height);
    gpuErrChk(cudaMemcpy(d_origin_image, h_origin_image, size_int, cudaMemcpyHostToDevice));
    dim3 block(32, 16,1);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y, 1);
    dim3 block_horizon(BLOCKDIMX, 16, 1);
    dim3 grid_horizon(1, (height+block_horizon.y-1)/block_horizon.y, 1);
    dim3 block_vertical(BLOCKDIMX, 16, 1);
    dim3 grid_vertical((width+block_vertical.x-1)/block_vertical.x, 1, 1);

    integral_phase1_kernel<<<grid, block>>>(d_origin_image, d_result_phase1, d_horizon_src, width, image_size);
    horizon_sum_kernel<<<grid_horizon, block_horizon>>>(d_horizon_src, d_horizon_dst, width, horizon_width);
    integral_phase2_kernel<<<grid, block>>>(d_result_phase1, d_result_phase2, d_horizon_dst, d_vertical_src,
                           width, horizon_width, vertical_width);
    vertical_sum_kernel<<<grid_vertical, block_vertical>>>(d_vertical_src, d_vertical_dst, width, vertical_width);
    integral_phase3_kernel<<<grid, block>>>(d_result_phase2, d_result_phase3, d_vertical_dst,
                           width, vertical_width);
    //gpuErrChk(cudaDeviceSynchronize());
    //auto end = getMoment;

    //cout << "Kernel time = "<< chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0  << endl;

    gpuErrChk(cudaMemcpy(h_result_phase3, d_result_phase3, size_int, cudaMemcpyDeviceToHost));
    //writeToFile("../variable/h_result_phase3.txt", h_result_phase3, width, height);

}

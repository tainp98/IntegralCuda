#ifndef INTEGRALIMAGE_H
#define INTEGRALIMAGE_H

#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "opencv2/opencv.hpp"
#include <fstream>
#include<opencv2/opencv.hpp>
#include <chrono>

#define HEIGHT 480
#define OVERLAP_WIDTH 80
#define getMoment std::chrono::high_resolution_clock::now()
using namespace std;
using namespace cv;

#define gpuErrChk(call) {gpuError((call));}
inline void gpuError(cudaError_t call){
    const cudaError_t error= call;
    if(error != cudaSuccess){
        printf("Error: %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}
inline void writeToFile(char * filename,int *data, int width, int height)
{
    printf("Writing to file ...\n");
    std::cout << filename << std::endl;
    FILE* file;
    file = fopen(filename, "w");
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            fprintf(file, "%d ", data[y * width + x]);
        }
        fprintf(file, "\n");
    }
//    fprintf(file, "%d ", data[width*height]);
//    fprintf(file, "%d ", data[width*height+1]);
    fclose(file);
}
class IntegralCuda
{
public:
    IntegralCuda();
    //IntegralCuda(int width, int height);

public:
    void h_mem_init();
    void d_mem_init();

    void cudaMemoryInit(int cols, int rows);
    void prefixSum2D();

    // De-allocates all the memory allocated on the host and the device
    void cudaCutsFreeMem();
    // Functions calculates the total energy of the configuration

public:
    //cv::Mat src, dst;
    /*************************************************
     * n-edges and t-edges                          **
     * **********************************************/
    int width, height, image_size, size_int;
    int *d_origin_image, *d_result_phase1, *d_result_phase2, *d_result_phase3;

    int *h_origin_image, *h_result_phase1, *h_result_phase2, *h_result_phase3;


};

#endif // INTEGRALIMAGE_H


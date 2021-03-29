#include <iostream>
#include "integralimage.h"
#include <bits/stdc++.h>
using namespace std;

#define R 128
#define C 640

// calculating new array
void prefixSum2D(int *a, int rows, int cols)
{

    // Filling first row and first column
    for (int i = 1; i < cols; i++)
        a[i] += a[i - 1];
    for (int i = 1; i < rows; i++)
        a[i*cols] += a[(i-1)*cols];

    // updating the values in the cells
    // as per the general formula
    for (int i = 1; i < rows; i++) {
        for (int j = 1; j < cols; j++)

            // values in the cells of new
            // array are updated
            a[i*cols+j] += a[(i-1)*cols+j] + a[i*cols+j-1]
                        - a[(i-1)*cols+j-1];
    }

    // displaying the values of the new array
//    for (int i = 0; i < rows; i++) {
//        for (int j = 0; j < cols; j++)
//            a[i*cols+j] = psa[i][j];

//    }
}
int checkResult(int *a, int *b, int rows, int cols){
    int sum = 0;
    for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++)
                sum = sum + a[i*cols+j] - b[i*cols+j];

        }
    return sum;
}

int main()
{
    cv::Mat img0 = cv::imread("/home/nvidia/imgs/images_filter/152_0.591195_1608895250645.png", cv::IMREAD_GRAYSCALE);
    cv::Mat D (img0, cv::Rect(35, 60, 640, 480) );
    cv::Mat m1;
    D.convertTo(m1, CV_32SC1);
    int *h_origin_image, *d_origin_image;
    h_origin_image = (int*)malloc(R*C*sizeof(int));
    gpuErrChk(cudaMalloc((void**)&d_origin_image, R*C*sizeof(int)));
    memcpy(h_origin_image, m1.ptr(0), R*C*sizeof(int));

//    for(int i=0; i<image_size; i++){
//        h_origin_image[i] = 1;
//    }
    //writeToFile("../variable/h_origin_image.txt", h_origin_image, width, height);
    gpuErrChk(cudaMemcpy(d_origin_image, h_origin_image, R*C*sizeof(int), cudaMemcpyHostToDevice));
    IntegralCuda a;
    a.cudaMemoryInit(640, 128);
    for(int i = 0; i < 20; i++)
        a.prefixSum2D(d_origin_image);
    auto start = getMoment;
    for(int i = 0; i < 100; i++)
        a.prefixSum2D(d_origin_image);

    auto end = getMoment;
    cout << "Optimize Time GPU = "<< chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0/100  << endl;
    int *cudaResult = a.h_result_phase3;
    cv::Mat isum;
    start = getMoment;
    cv::integral(D, isum,CV_64F);
    end = getMoment;
    cout << "opencv CPU time = "<< chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0  << endl;
    start = getMoment;
    for(int i = 1; i <= 100; i++)
        cv::cuda::integral(D, isum);
    end = getMoment;
    cout << "opencv cuda time = "<< chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0/100  << endl;
    cout << "stop 1 " << endl;
    int *arr;
    arr = (int*)malloc(R*C*sizeof(int));
    for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++)
                arr[i*C+j] = 1;

        }
//    cv::Mat E;
//    D.convertTo(E, CV_32SC1);
//    memcpy(arr, E.ptr(0), R*C*sizeof(int));
    cout << "stop 2 " << endl;
    start = getMoment;
    prefixSum2D(arr, R, C);
    end = getMoment;
    cout << "Optimize Time CPU = "<< chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0  << endl;

    cout << "stop 3 " << endl;
    cout << "checkResult " << checkResult(arr, cudaResult, R, C) << endl;
    a.cudaCutsFreeMem();
    //free(cudaResult);
    free(arr);

}


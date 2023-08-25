#include<iostream>
#include<Eigen\Dense>
#include"Tensor.cu"
#include<string>
#include<exception>
#include"BasicCalculateNode.cu"
#include<memory>
#include"ActivationFunction.cu"
#include"Module.cu"
#include"Optimizer.cu"
#include"LossFunction.cu"
#include<chrono>
#include"DatasetBase.cu"
#include"DataLoader.cu"
#include"cublas_v2.h"

// class myHandle{
// public:
//     int a;
//     static cublasHandle_t cublasH ;
// };

// cublasHandle_t myHandle::cublasH = nullptr;

// class InitHandle{
// public:
//     InitHandle(){
//         auto cudaInfo = cublasCreate(&myHandle::cublasH);
//         if(cudaInfo != CUBLAS_STATUS_SUCCESS){
//             std::cout << "can't get the cublas handle !" << std::endl;
//         }
//         else{
//             std::cout << "init handle okkkkk!";
//         }
//     }

//     ~InitHandle(){
//         auto cudaInfo = cublasDestroy(myHandle::cublasH);
//         if(cudaInfo != CUBLAS_STATUS_SUCCESS){
//             std::cout << "can't free the cublas handle !" << std::endl;
//         }
//         else{
//             std::cout << "free handle okkkkk!";
//         }
//     }
// };

// InitHandle init;
float mul(int left , int right )
{
    return -left * right;
}

int main(){
    int a = 2 , b = -1 ,c = 3 ,d = 1;
    float e,f;
    e = (a *c - mul(b,d))/(c*c - mul(d,d));
    f = (b * c - a * d)/(c*c - mul(d,d));
    std::cout << e << " " << f;
    return 0;
}
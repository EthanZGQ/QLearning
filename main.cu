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

int main(){

    Tensor<float> data({1,2,3,4} , true , nullptr , true);
     std::cout<< std::endl <<data.getData() << std::endl << std::endl; 
     std::cout<< std::endl <<data.getGrad() << std::endl << std::endl; 

    return 0;

}
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

    auto data = std::make_shared<Tensor<float>>(std::initializer_list<int>({2 , 2 , 4,4}));
    data->getData() << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
    std::cout << "the input data is " << std::endl<< data->getData() << std::endl << std::endl;
    int inputChannal = 2 , outputChannal = 7 , stride = 1, dilation = 0 , kernalSize = 3 , padding = 1 ;
    auto conv = std::make_shared<Conv2d<float>>(inputChannal , outputChannal , kernalSize , padding , stride , dilation);
    conv->preTensorNodes["weights"]->getData() = Eigen::ArrayXXf::Constant(outputChannal , kernalSize * kernalSize * inputChannal , 1);
    std::cout << "the weights data is " << std::endl<< conv->preTensorNodes["weights"]->getData() << std::endl << std::endl;
    auto output = conv->forward({data});
    std::cout << "the img2col data is " << std::endl<< conv->m_img2colData->getData() << std::endl << std::endl;
    std::cout << "the output data is " << std::endl << output->getData() << std::endl << std::endl;

    auto outputSize = output->getSize();
    auto col = output->shape().back();
    std::cout << "the outputSize is " << std::endl << outputSize << std::endl << std::endl;
    output->getGrad() = Eigen::ArrayXXf::Constant(col , outputSize/col , 1);
    std::cout << "the output grad is " << std::endl << output->getGrad() << std::endl << std::endl;
    output->backward();
    std::cout << "the input grad is " << std::endl << data->getGrad() << std::endl << std::endl;
    std::cout << "the weights grad is " << std::endl << conv->preTensorNodes["weights"]->getGrad() << std::endl << std::endl;



    return 0;
}
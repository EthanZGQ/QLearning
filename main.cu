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
    int inputChannal = 2 , outputChannal = 3 , stride = 1, dilation = 0 , kernalSize = 3 , padding = 1 ;
    auto conv = std::make_shared<Conv2d<float>>(inputChannal , outputChannal , kernalSize , padding , stride , dilation);
    std::cout << "the weights data is " << std::endl<< conv->preTensorNodes["weights"]->getData() << std::endl << std::endl;
    auto output = conv->forward({data});
    std::cout << "the img2col data is " << std::endl<< conv->m_img2colData->getData() << std::endl << std::endl;
    std::cout << "the output data is " << std::endl << output->getData() << std::endl << std::endl;
    // auto dataset = std::make_shared<constGenerator<float>>(22);
    // auto dataLoader = std::make_shared<DataLoader<float>>(dataset , 3 , 3);
    // for(int i = 0 ; i < 3 ; ++i){
    //     dataLoader->reset();    
    //     while(!dataLoader->empty()){
    //         auto data = dataLoader->getData();
    //         std::cout << "the input data is " << std::endl<< data.first->getData() << std::endl << std::endl;
    //         std::cout << "the input label is " << std::endl <<data.second->getData() << std::endl << std::endl;
    //         std::this_thread::sleep_for(std::chrono::seconds(5));
    //     }
    // }
    return 0;
}
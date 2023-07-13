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

    auto dataset = std::make_shared<constGenerator<float>>(22);
    auto dataLoader = std::make_shared<DataLoader<float>>(dataset , 3 , 3);
    for(int i = 0 ; i < 3 ; ++i){
        dataLoader->reset();    
        while(!dataLoader->empty()){
            auto data = dataLoader->getData();
            std::cout << "the input data is " << std::endl<< data.first->getData() << std::endl << std::endl;
            std::cout << "the input label is " << std::endl <<data.second->getData() << std::endl << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }
    return 0;
}
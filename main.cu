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
#include<unsupported\Eigen\CXX11\Tensor>



int main(){
    std::string dirName = "D:\\pytorch_dataset\\mnist\\train_images";
    auto dataset = std::make_shared<MnistDataset<float>>(dirName);
    auto dataLoader = std::make_shared<DataLoader<float>>(dataset , 3 , 4 , true);
    for(int i = 0 ; i < 1 ; ++i){
        dataLoader->reset();
        while(!dataLoader->empty()){
            auto dataAndLabel = dataLoader->getData();
            auto data = dataAndLabel.first;
            auto label = dataAndLabel.second;
            std::cout << "the data is " <<std::endl << data->getData().transpose()<< std::endl <<std::endl;
            std::cout << "the label is " <<std::endl << label->getData() << std::endl <<std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }


    return 0;
}
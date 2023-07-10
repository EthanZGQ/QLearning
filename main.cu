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


class MLP : public Module<float>{
public:
    MLP(){
        calculateNodeList["linear_1"] = std::make_shared<Linear<float>>(3,5);
        calculateNodeList["linear_2"] = std::make_shared<Linear<float>>(5,4);
        calculateNodeList["linear_3"] = std::make_shared<Linear<float>>(4,1);
        calculateNodeList["relu"] = std::make_shared<ReLu<float>>();
        calculateNodeList["sigmoid"] = std::make_shared<Sigmoid<float>>();
    }

    std::shared_ptr<Tensor<float>> forward(std::initializer_list<std::shared_ptr<Tensor<float>>> data) override{
        auto input = *(data.begin());
        

    }

}

int main(){

    

    return 0;
}
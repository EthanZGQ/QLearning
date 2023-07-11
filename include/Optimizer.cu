#ifndef OPTIMIZER
#define OPTIMIZER

#include"BasicCalculateNode.cu"
#include"Tensor.cu"
#include<memory>
#include<vector>
#include"Module.cu"


template<class T>
class Optimizer{
public:
    std::vector<std::shared_ptr<Tensor<T>>> learnParameterList;
    virtual void zeroGrad() = 0;
    virtual void step() = 0;
};


template<class T>
class SGD:public Optimizer<T>{
public:

    void zeroGrad() override {
        for(auto & data :  learnParameterList){
            data->zeroGrad();
        }
    }

    void step() override{
        for(auto & data : learnParameterList){
            data->adjust(learnRate);
        }
    }

    void findCalculateNode(std::shared_ptr<Module<T>> net){
        for(auto & x : net->calculateNodeList){
            for(auto & dataNode : x.second->preTensorNodes){
                if(dataNode.second->needGrad()){
                    learnParameterList.push_back(dataNode.second);
                }
            }
        }
        for(auto & x : net->moduleList){
            findCalculateNode(x.second);
        }
    }

    T learnRate ;
    SGD(std::shared_ptr<Module<T>> net , T lr = 0.001f){
        learnRate = lr;
        findCalculateNode(net);
    }
};




#endif
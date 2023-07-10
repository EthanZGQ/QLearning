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
    virtual zeroGrad() = 0;
    virtual step() = 0;
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

    void findCalculateNode(Module<T> & net){
        for(auto & x : net.calculateNodeList){
            for(auto & dataNode : x->preTensorNodes){
                if(dataNode->second->m_needGrad){
                    learnParameterList.push(dataNode->second);
                }
            }
        }
        for(auto & x : net.moduleList){
            findCalculateNode(x);
        }
    }

    T learnRate ;
    SGD(Module<T> & net , T lr = 0.001f){
        learnRate = lr;
        findCalculateNode(net);
    }
};




#endif
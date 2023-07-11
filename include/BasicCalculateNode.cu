#ifndef BASIC_CALCULATE_NODE
#define BASIC_CALCULATE_NODE

#include"CalculateNodeBase.cu"
#include<Tensor.cu>
#include<memory>
#include<initializer_list>

template<class T>
class MSELoss:public CalculateNodeBase<T>{
private:
    bool m_average = true;
    bool m_reduce = true;

public:
    
    MSELoss(bool average = true , bool reduce = true):m_average(average) , m_reduce(reduce) {};

    std::shared_ptr<Tensor<T>> forward(std::initializer_list<std::shared_ptr<Tensor<T>>> data) override{
        if(data.size() != 2) return nullptr;
        auto input = *(data.begin());
        auto label = *(data.begin() + 1);
        input->addUseTime();
        preTensorNodes["input"] = input;
        preTensorNodes["label"] = label;
        if(!m_reduce){
            if(!backTensorNode || backTensorNode->shape() != input->shape() ){
               backTensorNode = std::make_shared<Tensor<T>>(input->shape(),
               false , this );
            }
            backTensorNode->getData() = 1/2.f *(label->getData() - input->getData()) *(label->getData() - input->getData());
        }
        else{
            if(!backTensorNode || (backTensorNode->shape().size() != 1 && backTensorNode->shape()[0] != 1)){
               backTensorNode = std::make_shared<Tensor<T>>(std::initializer_list<int>{1 },false , this );
            }
            Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic> loss = 1/2.f *(label->getData() - input->getData()) *(label->getData() - input->getData()); 
            if(m_average){
                backTensorNode->getData() = loss.mean();
            }
            else{
                backTensorNode->getData() = loss.sum();
            }
        }
        return backTensorNode;

    } 

    void backward() override{
        preTensorNodes["input"]->getGrad() += -preTensorNodes["label"] ->getData() + preTensorNodes["input"] ->getData();
        if(m_average){
            preTensorNodes["input"]->getGrad() = preTensorNodes["input"]->getGrad()/backTensorNode->shape().front();
        }
    } 
};


template<class T>
class Linear:public CalculateNodeBase<T>{
private:
    int m_inFeature;
    int m_outFeature;
    bool m_bias = true;

    bool inferShape(std::initializer_list<std::shared_ptr<Tensor<T>>> & data){
        if(data.size() != 1) throw "Only need one input !";
        auto input = *(data.begin());
        if(input->shape().back() != m_inFeature) throw "The input last dim should same as the inFeature !";
        preTensorNodes["input"] = input;
        input->addUseTime();
        if(!backTensorNode){
            std::vector<int> tempSize = input->shape();
            tempSize.back() = m_outFeature;
            backTensorNode = std::make_shared<Tensor<T>>(tempSize , false , this);
        }
        else{
            std::vector<int> tempSize = input->shape();
            tempSize.back() = m_outFeature;
            if(tempSize != backTensorNode->shape()){
                backTensorNode = std::make_shared<Tensor<T>>(tempSize , false , this);
            }
        }
        return true;
    }

    void compute(){
        backTensorNode->getData() = preTensorNodes["weights"]->getData().matrix() * (preTensorNodes["input"]->getData()).matrix();
        if(m_bias){
            backTensorNode->getData().colwise() += preTensorNodes["bias"]->getData().rowwise().sum();
        }
    }


public:
    Linear(int in_feature , int out_feature , bool bias = true):m_inFeature(in_feature) , m_outFeature(out_feature) , m_bias(bias) {
        preTensorNodes["weights"] = std::make_shared<Tensor<T>>(std::initializer_list<int>({in_feature , out_feature}) , true);
        if(m_bias){
            preTensorNodes["bias"] = std::make_shared<Tensor<T>>(std::initializer_list<int>({1 , out_feature}) , true);
        }
    };

    std::shared_ptr<Tensor<T>> forward(std::initializer_list<std::shared_ptr<Tensor<T>>> data) override{
        bool checkOk = inferShape(data);
        compute();
        return backTensorNode;
    }

    void backward() override{
        preTensorNodes["weights"]->getGrad() = preTensorNodes["weights"]->getGrad().matrix() +  (backTensorNode->getGrad()).matrix() * (preTensorNodes["input"]->getData().transpose()).matrix();
        preTensorNodes["input"]->getGrad() = preTensorNodes["input"]->getGrad().matrix() +  (preTensorNodes["weights"]->getData().transpose()).matrix() * backTensorNode->getGrad().matrix();
        if(m_bias){
            preTensorNodes["bias"]->getGrad() = preTensorNodes["bias"]->getGrad()  +  backTensorNode->getGrad().rowwise().sum();
        }
    }

};


#endif //BASIC_CALCULATE_NODE
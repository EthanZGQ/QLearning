#ifndef LOSS_FUNCTION
#define LOSS_FUNCTION

#include"CalculateNodeBase.cu"
#include"Tensor.cu"

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
        label->addUseTime();
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
            preTensorNodes["input"]->getGrad() /= preTensorNodes["input"] ->getSize();
        }
    } 
};

template<class T>
class BCELoss:public CalculateNodeBase<T>{
private:
    bool m_average = true;
    bool m_reduce = true;

public:
    
    BCELoss(bool average = true , bool reduce = true):m_average(average) , m_reduce(reduce) {};

    std::shared_ptr<Tensor<T>> forward(std::initializer_list<std::shared_ptr<Tensor<T>>> data) override{
        if(data.size() != 2) return nullptr;
        auto input = *(data.begin());
        auto label = *(data.begin() + 1);
        input->addUseTime();
        label->addUseTime();
        preTensorNodes["input"] = input;
        preTensorNodes["label"] = label;
        if(!m_reduce){
            if(!backTensorNode || backTensorNode->shape() != input->shape() ){
               backTensorNode = std::make_shared<Tensor<T>>(input->shape(),
               false , this );
            }
            backTensorNode->getData() =-(label->getData()*input->getData().log() + (1 - label->getData())*(1 - input->getData()).log() ) ;
        }
        else{
            if(!backTensorNode || (backTensorNode->shape().size() != 1 && backTensorNode->shape()[0] != 1)){
               backTensorNode = std::make_shared<Tensor<T>>(std::initializer_list<int>{1 },false , this );
            }
            Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic> loss = -(label->getData()*input->getData().log() + (1 - label->getData())*(1 - input->getData()).log() ); 

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
        preTensorNodes["input"]->getGrad() += - preTensorNodes["label"]->getData() / preTensorNodes["input"]->getData() + (1 - preTensorNodes["label"]->getData()) / (1 - preTensorNodes["input"]->getData());
        if(m_average){
            preTensorNodes["input"]->getGrad() /= preTensorNodes["input"]->getSize();
        }
    } 
};
#endif
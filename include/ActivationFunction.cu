#ifndef ACTIVATION_FUNCTION
#define ACTIVATION_FUNCTION

#include"CalculateNodeBase.cu"
#include"Tensor.cu"
#include<initializer_list>
#include<memory>


template<typename T>
struct sigmoidForward{
__device__ T operator()(T & nums){
    return 1.0f / (1.0f + __expf(-nums));
}
};

template<typename T>
struct sigmoidBackward{
__device__ T operator()(T & afterGrad , T & value){
   return afterGrad * (1 - value) * value;
}
};

template<class T>
class Sigmoid:public CalculateNodeBase<T>{
private:
    bool preCheck(std::initializer_list<std::shared_ptr<Tensor<T>>> & data){
        if(data.size() != 1) throw "Need only one input !";
        auto input = * data.begin();
        preTensorNodes["input"] = input;
        input->addUseTime();

        if(!backTensorNode){
            backTensorNode = std::make_shared<Tensor<T>>(input->shape() , false , this);
        }
        else{
            if(backTensorNode->shape() != input->shape()){
                backTensorNode = std::make_shared<Tensor<T>>(input->shape() , false , this);
            }
        }
        return true;
    }
public:
    std::shared_ptr<Tensor<T>> forward(std::initializer_list<std::shared_ptr<Tensor<T>>> data) override{
        bool safe = preCheck(data);
        backTensorNode->getData() = 1.f/(1.f + preTensorNodes["input"]->getData().exp().cwiseInverse());
        return backTensorNode;
    }

    void backward() override{
        preTensorNodes["input"]->getGrad() += backTensorNode->getGrad() * (1.f - backTensorNode->getData()) * backTensorNode->getData();
    }

};



template<typename T>
struct reluForward{
__device__ T operator()(T & nums){
    T zero = static_cast<T>(0);
    return nums > zero ? nums : zero ;
}
};


template<typename T>
struct reluBackward{
__device__ T operator()(T & afterGrad , T & value){
    T zero = static_cast<T>(0);
    return value > zero ? afterGrad : zero ;
}
};

template<class T>
class ReLu:public CalculateNodeBase<T>{
private:
    bool preCheck(std::initializer_list<std::shared_ptr<Tensor<T>>> & data){
        if(data.size() != 1) throw "Need only one input !";
        auto input = * data.begin();
        preTensorNodes["input"] = input;
        input->addUseTime();

        if(!backTensorNode){
            backTensorNode = std::make_shared<Tensor<T>>(input->shape() , false , this);
        }
        else{
            if(backTensorNode->shape() != input->shape()){
                backTensorNode = std::make_shared<Tensor<T>>(input->shape() , false , this);
            }
        }
        return true;
    }
public:
    std::shared_ptr<Tensor<T>> forward(std::initializer_list<std::shared_ptr<Tensor<T>>> data) override{
        bool safe = preCheck(data);
        auto fun = [](T data){return data > 0? data : 0 ;};
        backTensorNode->getData() = preTensorNodes["input"]->getData().unaryExpr(fun);
        return backTensorNode;
    }

    void backward() override{
        auto fun = [](T first , T second){ return second > 0 ? first : 0 ;};
        preTensorNodes["input"]->getGrad() += backTensorNode->getGrad().binaryExpr(preTensorNodes["input"]->getData() ,fun);
    }

};

<<<<<<< HEAD
template<typename T>
struct tanhForward{
__device__ T operator()(T & nums){
    T exp_1 = __expf(nums);
    T exp_2 = static_cast<T>(1.0f)/exp_1;
    return (exp_1 - exp_2)/(exp_1 + exp_2);
}
};

template<typename T>
struct tanhBackward{
__device__ T operator()(T & afterGrad , T & value){
    return afterGrad * (static_cast<T>(1.0f) - value * value);
}
};



template<class T>
class Tanh:public CalculateNodeBase<T>{
private:
    bool preCheck(std::initializer_list<std::shared_ptr<Tensor<T>>> & data){
        if(data.size() != 1) throw "Need only one input !";
        auto input = * data.begin();
        preTensorNodes["input"] = input;
        input->addUseTime();

        if(!backTensorNode){
            backTensorNode = std::make_shared<Tensor<T>>(input->shape() , false , this);
        }
        else{
            if(backTensorNode->shape() != input->shape()){
                backTensorNode = std::make_shared<Tensor<T>>(input->shape() , false , this);
            }
        }
        return true;
    }
public:
    std::shared_ptr<Tensor<T>> forward(std::initializer_list<std::shared_ptr<Tensor<T>>> data) override{
        bool safe = preCheck(data);
        auto exp_1 = preTensorNodes["input"]->getData().exp();
        auto exp_2 = exp_1.cwiseInverse();
        backTensorNode->getData() = (exp_1 - exp_2) / (exp_1 + exp_2);
        return backTensorNode;
    }

    void backward() override{
        preTensorNodes["input"]->getGrad() += backTensorNode->getGrad() * (1 - backTensorNode->getData().pow(2));
    }

};




#endif
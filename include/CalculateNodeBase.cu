#ifndef CALCULATE_NODE
#define CALCULATE_NODE

#include<map>
#include<memory>
#include<string>
template<class T>
class Tensor;

template<class T>
class CalculateNodeBase{
public:
    std::map< std::string , std::shared_ptr<Tensor<T>>> preTensorNodes;
    std::shared_ptr<Tensor<T>> backTensorNode;
    virtual std::shared_ptr<Tensor<T>> forward(std::initializer_list<std::shared_ptr<Tensor<T>>> data) = 0;
    virtual void backward() = 0;
    // virtual std::shared_ptr<Tensor<T>> operator()(std::initializer_list<std::shared_ptr<Tensor<T>>> data) = 0;
};




#endif
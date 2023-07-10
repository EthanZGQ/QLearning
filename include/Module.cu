#ifndef MODULE
#define MODULE

#include"CalculateNodeBase.cu"
#include"Tensor.cu"
#include<map>
#include<memory>
#include<string>
#include<initializer_list>

template<class T>
class Module{
public:    
    std::map<std::string , std::shared_ptr<Module<T>>> moduleList;
    std::map<std::string , std::shared_ptr<calculateNodeBase<T>>> calculateNodeList;
    virtual std::shared_ptr<Tensor<T>> forward(std::initializer_list<std::shared_ptr<Tensor<T>>> data) = 0;
};



#endif
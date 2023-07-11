#ifndef DATASET_BASE
#define DATASET_BASE

#include"Tensor.cu"
#include<utility>
#include<memory>

template<class T>
class DatasetBase{
public:
    virtual int len() = 0;
    virtual std::pair<std::shared_ptr<Tensor<T>> , std::shared_ptr<Tensor<T>>> getItem(int idx) = 0;
};



#endif
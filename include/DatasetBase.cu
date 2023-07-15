#ifndef DATASET_BASE
#define DATASET_BASE

#include"Tensor.cu"
#include<utility>
#include<memory>
#include<random>
#include<Eigen/Dense>
#include<thread>

template<class T>
class DatasetBase{
public:
    virtual int len() = 0;
    virtual std::pair<std::shared_ptr<Tensor<T>> , std::shared_ptr<Tensor<T>>> getItem(int idx) = 0;
};


template<class T>
class constGenerator:public DatasetBase<T>{
private:
    int m_numData ;
public:
    constGenerator(int numData) :m_numData(numData){};

    int len() override{
        return m_numData;
    }

    std::pair<std::shared_ptr<Tensor<T>> , std::shared_ptr<Tensor<T>>> getItem(int idx) override{
        Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic> inputdata = Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic>::Constant(3,2 , idx);
        Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic> inputlabel = Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic>::Constant(2,1 , idx);
        auto inputDataTensor = std::make_shared<Tensor<T>>(std::initializer_list<int>({2,3}) , inputdata);
        auto inputLabelTensor = std::make_shared<Tensor<T>>(std::initializer_list<int>({1,2}) , inputlabel);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return {inputDataTensor, inputLabelTensor};
    }
};



#endif
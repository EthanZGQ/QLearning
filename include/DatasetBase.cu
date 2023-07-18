#ifndef DATASET_BASE
#define DATASET_BASE

#include"Tensor.cu"
#include<utility>
#include<memory>
#include<random>
#include<Eigen/Dense>
#include<thread>
#include<io.h>
#include<vector>
#include<string>
#include<opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

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


template<class T>
class MnistDataset:public DatasetBase<T>{
// private:
public:
    std::string m_dirPath;
    std::vector<std::string> m_filesName;
    void findFilesName(){
        intptr_t handle = 0;
        struct _finddata_t fileInfo;
        std::string wantFileName(m_dirPath);
        wantFileName.append("\\*.jpg");
        if((handle = _findfirst(wantFileName.c_str(), &fileInfo)) == -1) throw "this dir dont't have anything you want !";
        do{
            m_filesName.push_back(fileInfo.name);
        }while(_findnext(handle , & fileInfo) == 0);
        _findclose(handle);
    }

public:
    MnistDataset(const std::string& dirPath):m_dirPath(dirPath){
        findFilesName();
    }
    int len() override{
        return m_filesName.size();
    }

    std::pair<std::shared_ptr<Tensor<T>> , std::shared_ptr<Tensor<T>>> getItem(int idx) override{
        std::string fileName;
        fileName.assign(m_dirPath).append("\\").append(m_filesName[idx]);
        cv::Mat img = cv::imread(fileName , 0);
        img.resize(28);
        Eigen::Matrix<T , Eigen::Dynamic , Eigen::Dynamic> data;
        cv::cv2eigen(img , data);
        auto inputDataTensor = std::make_shared<Tensor<T>>(std::initializer_list<int>({28,28}));
        inputDataTensor->getData() = (data.transpose().array());
        Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic> label(10,1);
        label.setZero();
        int labelIdx = *(fileName.end() - 5) - '0';
        label(labelIdx , 0) = 1;
        auto inputLabelTensor = std::make_shared<Tensor<T>>(std::initializer_list<int>({1,10}) , label);
        return {inputDataTensor , inputLabelTensor};
    }


};


#endif
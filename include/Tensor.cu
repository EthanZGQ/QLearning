#ifndef TENSOR
#define TENSOR
#include"CalculateNodeBase.cu"
#include<Eigen\Dense>
#include<vector>
#include<iostream>
#include<queue>
#include"cuda.h"
#include"cuda_runtime.h"
#include"cublas_v2.h"



template<class T>
class Tensor{
private:
    Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic> m_cpuData;
    Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic> m_cpuGrad;
    T * m_gpuData = nullptr;
    T * m_gpuGrad = nullptr;
    std::vector<int> m_shape;
    bool m_needGrad = false;
    bool m_inCuda;
    CalculateNodeBase<T> *m_preCalculateNode = nullptr;
    int m_useTime = 0;
    int m_size = 0;

public:
    
    Tensor(std::vector<int> size , bool needGrad = false ,CalculateNodeBase<T> *fatherNode = nullptr , bool useCuda = false){
        m_shape = size;
        int rows = 1;
        for(auto & x : m_shape){
            if(!x) throw "The size number should greater than 0 !";
            rows *= x;
        }
        m_size = rows;
        rows /= m_shape.back();
        m_needGrad = needGrad;
        m_inCuda = useCuda;
        if(typeid(T) != typeid(float)  && typeid(T) != typeid(int) && typeid(T) != typeid(double)){
            throw "only support float int double ! I'm so sorry !";
        }
        if(!m_inCuda){
            m_cpuData = Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic>::Random(m_shape.back() , rows);
            if(m_needGrad)
            m_cpuGrad = Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic>::Zero(m_shape.back() , rows);
        }
        else{
            auto cudaInfo = cudaMalloc((void **) & m_gpuData , sizeof(T) * m_size);
            if(cudaInfo != cudaSuccess){
                throw "can't cudaMalloc the gpuData !";
            } 
            if(m_needGrad){
                cudaInfo = cudaMalloc((void **) & m_gpuGrad , sizeof(T) * m_size);
                if(cudaInfo != cudaSuccess){
                    throw "can't cudaMalloc the gpuGrad !";
                } 
            }
        }
        if(m_gpuData) std::cout << "get the gpu data mem !";
        if(m_gpuGrad) std::cout << "get the gpu grad mem !";
        m_preCalculateNode = fatherNode;
    }

    Tensor(std::initializer_list<int> size , Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic> & data , 
    bool needGrad = false , CalculateNodeBase<T> * fatherNode = nullptr){
        m_shape = size ;
        if(data.rows() != m_shape.back()){
            throw "input data last dim should same sa the size last dim !";
        }
        int tempSize = 1;
        for(auto x : m_shape){
            tempSize *= x;
            if(!x) throw "The size number should greater than 0 !";
        }
        if(data.cols() * data.rows() != tempSize){
            throw "input data size should same as input size";
        }
        m_size = tempSize;
        m_needGrad = needGrad;
        m_cpuData = data;
        if(m_needGrad){
            m_cpuGrad = Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic>::Zero(data.rows() , data.cols());
        }
        m_preCalculateNode = fatherNode;
    }

    void cuda(){
        if(m_inCuda) return;
        m_inCuda = true;
        cudaError_t cudaInfo;
        if(m_gpuData == nullptr){
            auto cudaInfo = cudaMalloc((void **) & m_gpuData , sizeof(T) * m_size);
            if(cudaInfo != cudaSuccess){
                std::cout << "to cuda ,cudaMalloc m_gpuData failed !";
                throw  "to cuda ,cudaMalloc m_gpuData failed !";
            }
        }

        cudaInfo = cudaMemcpy(m_gpuData , m_cpuData.data() , m_size * sizeof(T) , cudaMemcpyHostToDevice);
        if(cudaInfo != cudaSuccess){
            std::cout << "to cuda ,cudaMemcpy m_gpuData failed !";
            throw  "to cuda ,cudaMemcpy m_gpuData failed !";
        }

        if(m_needGrad){
            if(m_gpuGrad == nullptr){
                auto cudaInfo = cudaMalloc((void **) & m_gpuGrad , sizeof(T) * m_size);
                if(cudaInfo != cudaSuccess){
                    std::cout << "to cuda ,cudaMalloc m_gpuGrad failed !";
                    throw  "to cuda ,cudaMalloc m_gpuGrad failed !";
                }
            }
            cudaInfo = cudaMemcpy(m_gpuGrad , m_cpuGrad.data() , m_size * sizeof(T) , cudaMemcpyHostToDevice);
            if(cudaInfo != cudaSuccess){
                std::cout << "to cuda ,cudaMemcpy m_gpuGrad failed !";
                throw  "to cuda ,cudaMemcpy m_gpuGrad failed !";
            }
        }
    }

    void host(){
        if(!m_inCuda) return ;
        if(m_cpuData.rows() == 0){
            m_cpuData = Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic>(m_shape.back() , m_size / m_shape.back());
        }
        if(m_needGrad && m_cpuGrad.rows() == 0){
            m_cpuGrad = Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic>(m_shape.back() , m_size / m_shape.back());
        }
        auto cudaInfo = cudaMemcpy(m_cpuData.data() ,m_gpuData , m_size * sizeof(T) , cudaMemcpyDeviceToHost);
        if(cudaInfo != cudaSuccess){
            std::cout << "to host ,cudaMemcpy m_gpuData failed !";
            throw  "to host ,cudaMemcpy m_gpuData failed !";
        }
        if(m_gpuGrad){
            cudaInfo = cudaMemcpy(m_cpuGrad.data() , m_gpuGrad , m_size * sizeof(T) , cudaMemcpyHostToDevice);
            if(cudaInfo != cudaSuccess){
                std::cout << "to host ,cudaMemcpy m_gpuGrad failed !";
                throw  "to host ,cudaMemcpy m_gpuGrad failed !";
            }
        }
        m_inCuda = false;
    }

    Eigen::Array<T,Eigen::Dynamic , Eigen::Dynamic> & getData(){
        if(m_inCuda){
            if(m_cpuData.rows() == 0){
                m_cpuData = Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic>(m_shape.back() , m_size / m_shape.back());
            }
            auto cudaInfo = cudaMemcpy(m_cpuData.data() ,m_gpuData , m_size * sizeof(T) , cudaMemcpyDeviceToHost);
            if(cudaInfo != cudaSuccess){
                std::cout << "to host ,cudaMemcpy m_gpuData failed !";
                throw  "to host ,cudaMemcpy m_gpuData failed !";
            }
        }
        return m_cpuData;
    }

    Eigen::Array<T,Eigen::Dynamic , Eigen::Dynamic> & getGrad(){
        if(m_inCuda){
            if(m_cpuGrad.rows() == 0){
                m_cpuGrad = Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic>(m_shape.back() , m_size / m_shape.back());
            }
            auto cudaInfo = cudaMemcpy(m_cpuGrad.data() ,m_gpuGrad , m_size * sizeof(T) , cudaMemcpyDeviceToHost);
            if(cudaInfo != cudaSuccess){
                std::cout << "to host ,cudaMemcpy m_gpuGrad failed !";
                throw  "to host ,cudaMemcpy m_gpuGrad failed !";
            }
        }
        return m_cpuGrad;
    }


    void backward(){
        std::queue<CalculateNodeBase<T>*> myQ;
        myQ.push(m_preCalculateNode);
        while(myQ.size()){
            auto calNode = myQ.front();
            myQ.pop();
            calNode->backward();
            calNode->backTensorNode->zeroGrad();
            for(auto & x : calNode->preTensorNodes){
                auto ptr = x.second;
                ptr->subUseTime();
                auto tempPtr = ptr->getPreCalculateNode();
                if( ptr->useTimeEmpty() && tempPtr) myQ.push(tempPtr);
            }
        }
    }
 
    ~Tensor(){
        if(m_gpuData) {
            std::cout << "delete data  okkkk ";
            cudaFree(m_gpuData);
            }
        if(m_gpuGrad){
            std::cout << "delete grad  okkkk ";
            cudaFree(m_gpuGrad);
        } 
    }

    T * getCudaDataPtr(){
        return m_gpuData;
    }

    T * getCudaGradPtr(){
        return m_gpuGrad;
    }

    bool inCuda(){
        return m_inCuda;
    }

    void adjust(T lr){
        m_cpuData -= m_cpuGrad * lr;
    }

    void zeroGrad(){
        m_cpuGrad.setZero();
    }

    bool useTimeEmpty(){
        return m_useTime == 0;
    }

    void addUseTime(){
        ++m_useTime;
    }

    void subUseTime(){
        --m_useTime;
    }

    int getUseTime(){
        return m_useTime;
    }
    int getSize(){
        return m_size;
    }

    CalculateNodeBase<T> * getPreCalculateNode(){ 
        return m_preCalculateNode;
    }

    const std::vector<int> & shape(){
        return m_shape;
    }

    bool needGrad(){
        return m_needGrad;
    }

};


#endif
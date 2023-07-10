#ifndef TENSOR
#define TENSOR
#include"CalculateNodeBase.cu"
#include<Eigen\Dense>
#include<vector>
#include<iostream>
#include<queue>

template<class T>
class Tensor{
private:
    Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic> m_data;
    Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic> m_grad;
    std::vector<int> m_shape;
    // std::shared_ptr<T> m_data = nullptr;
    // std::shared_ptr<T> m_grad = nullptr;
    bool m_needGrad = false;
    CalculateNodeBase<T> *m_preCalculateNode = nullptr;
    int useTime = 0;
    int m_size = 0;

public:

    Tensor(std::initializer_list<int> size , bool needGrad = false , CalculateNodeBase<T> *fatherNode = nullptr){
        m_shape = size;
        int rows = 1;
        for(auto & x : m_shape){
            if(!x) throw "The size number should greater than 0 !";
            rows *= x;
        }
        m_size = rows;
        rows /= m_shape.back();
        m_needGrad = needGrad;
        if(typeid(T) == typeid(float)  || typeid(T) == typeid(int) || typeid(T) == typeid(double)){
            m_data = Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic>::Random(m_shape.back() , rows);
            m_grad = Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic>::Zero(m_shape.back() , rows);
        }
        else{
            throw "only support float int double ! I'm so sorry !";
        }
        m_preCalculateNode = fatherNode;
    }   
    
    Tensor(std::vector<int> size , bool needGrad = false , CalculateNodeBase<T> *fatherNode = nullptr){
        m_shape = size;
        int rows = 1;
        for(auto & x : m_shape){
            if(!x) throw "The size number should greater than 0 !";
            rows *= x;
        }
        m_size = rows;
        rows /= m_shape.back();
        m_needGrad = needGrad;
        if(typeid(T) == typeid(float)  || typeid(T) == typeid(int) || typeid(T) == typeid(double)){
            m_data = Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic>::Random(m_shape.back() , rows);
            m_grad = Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic>::Zero(m_shape.back() , rows);
        }
        else{
            throw "only support float int double ! I'm so sorry !";
        }
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
        m_data = data;
        m_grad = Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic>::Zero(data.rows() , data.cols());
        m_needGrad = needGrad;
        m_preCalculateNode = fatherNode;
        
    }



    Eigen::Array<T,Eigen::Dynamic , Eigen::Dynamic> & getData(){
        return m_data;
    }

    Eigen::Array<T,Eigen::Dynamic , Eigen::Dynamic> & getGrad(){
        return m_grad;
    }


    void backward(){
        std::cout <<"backward begin" <<std::endl;
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
        std::cout << "backward okkkk" << std::endl;
    }

    void adjust(){
        m_data -= m_grad * 0.01f;
    }

    void zeroGrad(){
        m_grad.setZero();
    }

    bool useTimeEmpty(){
        return useTime == 0;
    }

    void addUseTime(){
        ++useTime;
    }

    void subUseTime(){
        --useTime;
    }

    int getUseTime(){
        return useTime;
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

};


#endif
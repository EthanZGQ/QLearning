#ifndef BASIC_CALCULATE_NODE
#define BASIC_CALCULATE_NODE

#include"CalculateNodeBase.cu"
#include<Tensor.cu>
#include<memory>
#include<initializer_list>


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


template<class T>
class Conv2d :public CalculateNodeBase<T>{
public:
    int m_inChannals;
    int m_outChannals;
    int m_kernalSize;
    int m_stride;
    int m_padding;
    int m_dilation;
    std::shared_ptr<Tensor<T>> m_img2colData = nullptr;

    bool inferShape(std::initializer_list<std::shared_ptr<Tensor<T>>> & data){
        if(data.size() != 1) throw "Only need one input !";
        auto input = *(data.begin());
        if(input->shape().size() != 4) throw "The conv2d need shape 4 dim such as (batch,channal,height,width)";
        if(input->shape()[1]!= m_inChannals ) throw "The input second dim should same as the inChannals !";
        preTensorNodes["input"] = input;
        input->addUseTime();
        prepareImg2colData();
    }

    void prepareImg2colData(){
        auto inputShape = preTensorNodes["input"]->shape();
        int batch = inputShape.front();
        int height = inputShape[2];
        int width = inputShape.back();

        int colTime = std::ceilf((width + 2 * m_padding - m_kernalSize - (m_kernalSize - 1) *m_dilation + 1)
        /static_cast<float>(m_stride)); // 每行进行了几次卷积
        int rowTime = std::ceilf((height + 2 * m_padding - m_kernalSize - (m_kernalSize - 1) *m_dilation + 1)
        /static_cast<float>(m_stride)); // 每行进行了几次卷积
        int matrixRow = m_kernalSize * m_kernalSize * m_inChannals;
        int matrixCol = colTime * rowTime * batch;
        if(m_img2colData == nullptr){
            m_img2colData = std::make_shared<Tensor<T>>(std::initializer_list<int>({matrixCol , matrixRow}));
            backTensorNode = std::make_shared<Tensor<T>>(std::initializer_list<int>({batch , m_outChannals , rowTime , colTime}) , false , this);
        }
        else{
            std::vector<int> nowShape = {batch , m_outChannals , rowTime , colTime};
            if(nowShape != backTensorNode->shape()){
                m_img2colData = std::make_shared<Tensor<T>>(std::initializer_list<int>({matrixCol , matrixRow}));
                backTensorNode = std::make_shared<Tensor<T>>(nowShape , false , this);
            }
        }
    }


    void compute(){
        img2colCpu();
        Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic> value = preTensorNodes["weights"]->getData().matrix() * m_img2colData->getData().matrix();
        std::cout << "the output data is " << std::endl << value << std::endl << std::endl;
        int batch = backTensorNode->shape().front();
        int lineLen = value.size() / (m_outChannals * batch);
        int colLen = backTensorNode->shape().back();
        int rowLen = backTensorNode->shape()[2];
        for(int tempBatch = 0 ; tempBatch < batch ; ++tempBatch){
            Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic> tempValue = value.block(0 , lineLen * tempBatch , m_outChannals , lineLen);
            tempValue.transposeInPlace();
            tempValue.resize(colLen , rowLen*m_outChannals);
            backTensorNode->getData().block(0 , tempBatch * m_outChannals * rowLen , colLen , m_outChannals * rowLen) = tempValue;
        }

    }

    void backwardCompute(){
        int batch = backTensorNode->shape().front();
        int colLen = backTensorNode->shape().back();
        int rowLen = backTensorNode->shape()[2];
        int lineLen = colLen * rowLen;
        Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic> backNodeGrad(m_outChannals , batch * lineLen);
        for(int tempBatch = 0 ; tempBatch < batch ; ++tempBatch){
            Eigen::Array<T , Eigen::Dynamic , Eigen::Dynamic> tempValue = backTensorNode->getGrad().block(0 , m_outChannals * rowLen* tempBatch , colLen , m_outChannals * rowLen);
            tempValue.resize(lineLen , m_outChannals);
            tempValue.transposeInPlace();
            backNodeGrad.block(0 , tempBatch * lineLen , m_outChannals , lineLen) = tempValue;
        }
        preTensorNodes["weights"]->getGrad() += (backNodeGrad.matrix() * m_img2colData->getData().transpose().matrix()).array();
        m_img2colData->getGrad() += (preTensorNodes["weights"]->getData().transpose().matrix() * backNodeGrad.matrix()).array();
        img2colbackwardCpu();
        m_img2colData->getGrad().setZero();
    }

    void img2colbackwardCpu(){
        auto inputShape = preTensorNodes["input"]->shape();
        int batch = inputShape[0] , height = inputShape[2] , width = inputShape[3];
        int colTime = backTensorNode->shape().back() , rowTime = backTensorNode->shape()[2];
        int imgSize = width * height; //一个图片的大小
        int featureMapSize = m_inChannals * imgSize; //一个特征图的大小
        int flatKernalSize = m_kernalSize * m_kernalSize; //一个单层卷积核的大小
        int oneLineSize = flatKernalSize * m_inChannals; //img2col之后 一列的长度
        int oneLayerSize = oneLineSize* colTime * rowTime; //一个特征图 img2col之后的内存大小
        int realKernalSize = m_kernalSize + (m_kernalSize - 1) * m_dilation; //经过稀疏卷积后的卷积和的宽高长度
        T * output = preTensorNodes["input"]->getGrad().data();
        T * input = m_img2colData->getGrad().data();
        for(int _batch = 0 ; _batch < batch ; ++_batch){
            for(int row = -m_padding ; row <= height + m_padding - realKernalSize; row += m_stride){
                for(int col = -m_padding ; col <= width + m_padding - realKernalSize; col += m_stride){
                    for(int _feature = 0 ; _feature < m_inChannals ; ++ _feature){
                        for(int y = 0 ; y < m_kernalSize ; ++y){
                            for(int x = 0 ; x < m_kernalSize ; ++x){
                                T value ;
                                int realY = y*(m_dilation + 1);
                                int realX = x*(m_dilation + 1);
                                if(realY + row < 0 || realY + row >= height || realX + col < 0 || realX + col >= width){ //判断边界关系
                                    continue;
                                } 
                                else {
                                    int index = _batch * oneLayerSize + ((row + m_padding)/m_stride) * colTime *oneLineSize + 
                                    (col+m_padding)/m_stride * oneLineSize + flatKernalSize * _feature + y*m_kernalSize + x ;
                                    value = input[index]; 
                                }                //一个特征图的内存长度        //一行之后的内存长度       //一行之中的内存长度  
                                int imgIndex = _batch * featureMapSize + _feature * imgSize + (realY + row)*width + (realX + col);
                                output[imgIndex] += value;
                            }
                        }
                    }
                }
            }
        }
    }

    void img2colCpu(){
        auto inputShape = preTensorNodes["input"]->shape();
        int batch = inputShape[0] , height = inputShape[2] , width = inputShape[3];
        int colTime = backTensorNode->shape().back() , rowTime = backTensorNode->shape()[2];
        int imgSize = width * height; //一个图片的大小
        int featureMapSize = m_inChannals * imgSize; //一个特征图的大小
        int flatKernalSize = m_kernalSize * m_kernalSize; //一个单层卷积核的大小
        int oneLineSize = flatKernalSize * m_inChannals; //img2col之后 一列的长度
        int oneLayerSize = oneLineSize* colTime * rowTime; //一个特征图 img2col之后的内存大小
        int realKernalSize = m_kernalSize + (m_kernalSize - 1) * m_dilation; //经过稀疏卷积后的卷积和的宽高长度
        T * input = preTensorNodes["input"]->getData().data();
        T * output = m_img2colData->getData().data();
        for(int _batch = 0 ; _batch < batch ; ++_batch){
            for(int row = -m_padding ; row <= height + m_padding - realKernalSize; row += m_stride){
                for(int col = -m_padding ; col <= width + m_padding - realKernalSize; col += m_stride){
                    for(int _feature = 0 ; _feature < m_inChannals ; ++ _feature){
                        for(int y = 0 ; y < m_kernalSize ; ++y){
                            for(int x = 0 ; x < m_kernalSize ; ++x){
                                T value ;
                                int realY = y*(m_dilation + 1);
                                int realX = x*(m_dilation + 1);
                                if(realY + row < 0 || realY + row >= height || realX + col < 0 || realX + col >= width){ //判断边界关系
                                    value = 0;
                                } 
                                else {
                                    int imgIndex = _batch * featureMapSize + _feature * imgSize + (realY + row)*width + (realX + col);
                                    value = input[imgIndex];
                                }                //一个特征图的内存长度        //一行之后的内存长度       //一行之中的内存长度  
                                int index = _batch * oneLayerSize + ((row + m_padding)/m_stride) * colTime *oneLineSize + 
                                (col+m_padding)/m_stride * oneLineSize + flatKernalSize * _feature + y*m_kernalSize + x ;
                                output[index] = value; 
                            }
                        }
                    }
                }
            }
        }
    }

// public:
    Conv2d(int inChannals , int outChannals , int kernalSize , int padding = 0 , int stride = 1 , int dilation = 0):
    m_inChannals(inChannals) , m_outChannals(outChannals) , m_kernalSize(kernalSize) , m_padding(padding) , m_stride(stride) , m_dilation(dilation){
        preTensorNodes["weights"] = std::make_shared<Tensor<T>>(std::initializer_list<int>({ m_inChannals, m_kernalSize , m_kernalSize , m_outChannals}) , true );
        preTensorNodes["weights"]->getData().setConstant(1);
    }
    std::shared_ptr<Tensor<T>> forward(std::initializer_list<std::shared_ptr<Tensor<T>>> data) override{
        inferShape(data);
        compute(); 
        return backTensorNode;
    }

    void backward(){
        backwardCompute();
    };

};


#endif //BASIC_CALCULATE_NODE
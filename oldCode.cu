#include<iostream>
#include<Eigen\Dense>
#include"Tensor.cu"
#include<string>
#include<exception>
#include"BasicCalculateNode.cu"
#include<memory>
#include"ActivationFunction.cu"
#include"Module.cu"
#include"Optimizer.cu"
#include"LossFunction.cu"
#include<chrono>


class MLP : public Module<float>{
public:
    MLP(){
        calculateNodeList["linear_1"] = std::make_shared<Linear<float>>(3,30);
        calculateNodeList["linear_2"] = std::make_shared<Linear<float>>(30,20);
        calculateNodeList["linear_3"] = std::make_shared<Linear<float>>(20,1);
        calculateNodeList["relu"] = std::make_shared<ReLu<float>>();
        calculateNodeList["sigmoid"] = std::make_shared<ReLu<float>>();
    }

    std::shared_ptr<Tensor<float>> forward(std::initializer_list<std::shared_ptr<Tensor<float>>> data) override{
        auto input = *(data.begin());
        auto output_1 = calculateNodeList["linear_1"]->forward({input});
        auto active_1 = calculateNodeList["relu"]->forward({output_1});
        auto output_2 = calculateNodeList["linear_2"]->forward({active_1});
        auto active_2 = calculateNodeList["sigmoid"]->forward({output_2});
        auto output_3 = calculateNodeList["linear_3"]->forward({active_2});
        return output_3;
    }

};

int main(){

    auto mlp = std::make_shared<MLP>();
    auto opt = std::make_shared<SGD<float>>(mlp , 0.001f);
    auto bceLoss = std::make_shared<MSELoss<float>>();
    int turn = 6000;
    std::srand(std::chrono::system_clock::now().time_since_epoch().count());
    for(int i = 0 ; i < turn ; ++i){
        opt->zeroGrad();
        Eigen::ArrayXXf input(3,2);
        std::vector<float> a(3) , b(3);
        for(int i = 0 ; i < 3 ;++i){
            a[i] = std::rand()%20000 * 0.0003f;
            b[i] = std::rand()%20000 * 0.0003f;
        }
        input << a[0] , b[0] , a[1] , b[1] , a[2] , b[2];
        auto inputTensor = std::make_shared<Tensor<float>>(std::initializer_list<int>({2,3}) , input);
        Eigen::ArrayXXf label(1 ,2);
        float aSum = 0 , bSum = 0;
        for(int i = 0 ; i < 3 ; ++i){
            aSum += a[i] * a[i];
            bSum += b[i] * b[i]; 
        }
        label << aSum , bSum;
        auto labelTensor = std::make_shared<Tensor<float>>(std::initializer_list<int>({2 , 1}) , label);
        // if(i >= turn - 5)
        // std::cout << "input is " << std::endl << inputTensor->getData() << std::endl << std::endl;
        // std::cout <<" the optomizer get prameters num is " << opt->learnParameterList.size() << std::endl;
        auto ans = mlp->forward({inputTensor});
        if(i >= turn - 20){
        std::cout << "output is " << std::endl << ans->getData() << std::endl << std::endl;

        std::cout << "label is " << std::endl << labelTensor->getData() << std::endl << std::endl;
        }
        auto loss = bceLoss->forward({ans , labelTensor});
        std::cout << "loss is " << std::endl << loss->getData() << std::endl << std::endl;

        loss->backward();
        opt->step();

    }
    // Eigen::ArrayXXf input = Eigen::ArrayXXf::Constant(3,2 , 0.9798);
    // auto inputTensor = std::make_shared<Tensor<float>>(std::initializer_list<int>({2,3}) , input);
    // std::cout << "input is " << std::endl << inputTensor->getData() << std::endl << std::endl;
    // std::cout <<" the optomizer get prameters num is " << opt->learnParameterList.size() << std::endl;
    // std::cout << "output is " << std::endl << ans->getData() << std::endl << std::endl;

    // Eigen::ArrayXXf label = Eigen::ArrayXXf::Constant(3 , 2 , 0);
    // auto labelTensor = std::make_shared<Tensor<float>>(std::initializer_list<int>({2 , 3}) , label);
    // std::cout << "label is " << std::endl << labelTensor->getData() << std::endl << std::endl;
    // auto bceLoss = std::make_shared<BCELoss<float>>();
    // auto loss = bceLoss->forward({inputTensor , labelTensor});
    // std::cout << "loss is " << std::endl << loss->getData() << std::endl << std::endl;

    // loss->backward();

    // std::cout << "grad is " << std::endl << inputTensor->getGrad() << std::endl << std::endl;

    return 0;
}




    // auto dataset = std::make_shared<constGenerator<float>>(22);
    // auto dataLoader = std::make_shared<DataLoader<float>>(dataset , 3 , 3);
    // for(int i = 0 ; i < 3 ; ++i){
    //     dataLoader->reset();    
    //     while(!dataLoader->empty()){
    //         auto data = dataLoader->getData();
    //         std::cout << "the input data is " << std::endl<< data.first->getData() << std::endl << std::endl;
    //         std::cout << "the input label is " << std::endl <<data.second->getData() << std::endl << std::endl;
    //         std::this_thread::sleep_for(std::chrono::seconds(5));
    //     }
    // }


 struct _finddata_t fileInfo;
    
    Eigen::Tensor<float , 4> temsorData(1,4,5,5);
    temsorData.setConstant(1);
    std::cout << "the temsorData data is " << std::endl<< temsorData << std::endl << std::endl;
    auto data = std::make_shared<Tensor<float>>(std::initializer_list<int>({2 , 2 , 4,4}));
    data->getData() << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
    std::cout << "the input data is " << std::endl<< data->getData() << std::endl << std::endl;

    auto flatten = std::make_shared<Flatten<float>>();
    auto output = flatten->forward({data});
    std::cout << "the output data is " << std::endl<< output->getData() << std::endl << std::endl;
    float low = 0 , hight = 64;
    output->getGrad().setRandom();
    std::cout << "the output data is " << std::endl<< output->getGrad() << std::endl << std::endl;
    output->backward();
    std::cout << "the input grad is " << std::endl<< data->getGrad() << std::endl << std::endl;
    // int inputChannal = 2 , outputChannal = 7 , stride = 1, dilation = 0 , kernalSize = 3 , padding = 1 ;
    // auto conv = std::make_shared<Conv2d<float>>(inputChannal , outputChannal , kernalSize , padding , stride , dilation);
    // conv->preTensorNodes["weights"]->getData() = Eigen::ArrayXXf::Constant(outputChannal , kernalSize * kernalSize * inputChannal , 1);
    // std::cout << "the weights data is " << std::endl<< conv->preTensorNodes["weights"]->getData() << std::endl << std::endl;
    // auto output = conv->forward({data});
    // std::cout << "the img2col data is " << std::endl<< conv->m_img2colData->getData() << std::endl << std::endl;
    // std::cout << "the output data is " << std::endl << output->getData() << std::endl << std::endl;

    // auto outputSize = output->getSize();
    // auto col = output->shape().back();
    // std::cout << "the outputSize is " << std::endl << outputSize << std::endl << std::endl;
    // output->getGrad() = Eigen::ArrayXXf::Constant(col , outputSize/col , 1);
    // std::cout << "the output grad is " << std::endl << output->getGrad() << std::endl << std::endl;
    // output->backward();
    // std::cout << "the input grad is " << std::endl << data->getGrad() << std::endl << std::endl;
    // std::cout << "the weights grad is " << std::endl << conv->preTensorNodes["weights"]->getGrad() << std::endl << std::endl;




class LeNet :public Module<float>{
public:
    LeNet(){
        calculateNodeList["conv1"] = std::make_shared<Conv2d<float>>(1 , 6 , 3 , 1);
        calculateNodeList["conv2"] = std::make_shared<Conv2d<float>>(6 , 18 , 2 , 0 , 2);
        calculateNodeList["conv3"] = std::make_shared<Conv2d<float>>(18 , 12 , 3 , 1);
        calculateNodeList["conv4"] = std::make_shared<Conv2d<float>>(12 , 12 , 2 , 0 , 2); 
        calculateNodeList["relu1"] = std::make_shared<ReLu<float>>();
        calculateNodeList["relu2"] = std::make_shared<ReLu<float>>();
        calculateNodeList["relu3"] = std::make_shared<ReLu<float>>();
        calculateNodeList["relu4"] = std::make_shared<ReLu<float>>();
        calculateNodeList["flatten"] = std::make_shared<Flatten<float>>();
        calculateNodeList["linear1"] = std::make_shared<Linear<float>>(7 * 7 * 12 , 64);
        calculateNodeList["relu5"] = std::make_shared<Sigmoid<float>>();
        calculateNodeList["linear2"] = std::make_shared<Linear<float>>(64 , 32);
        calculateNodeList["relu6"] = std::make_shared<Sigmoid<float>>();
        calculateNodeList["linear3"] = std::make_shared<Linear<float>>(32 , 1);
        calculateNodeList["sigmoid"] = std::make_shared<Sigmoid<float>>();
    }

    std::shared_ptr<Tensor<float>> forward(std::initializer_list<std::shared_ptr<Tensor<float>>> data) override{
        auto input = *(data.begin());
        // std::cout <<"forward begin !" << std::endl;
        auto out_1 = calculateNodeList["conv1"]->forward({input});
        auto out_2 = calculateNodeList["relu1"]->forward({out_1});

        // std::cout <<"can forward conv1" << std::endl;

        auto out_3 = calculateNodeList["conv2"]->forward({out_2});
        auto out_4 = calculateNodeList["relu2"]->forward({out_3});


        // std::cout <<"can forward conv2" << std::endl;

        auto out_5 = calculateNodeList["conv3"]->forward({out_4});
        auto out_6 = calculateNodeList["relu3"]->forward({out_5});

        // std::cout <<"can forward conv3" << std::endl;


        auto out_7 = calculateNodeList["conv4"]->forward({out_6});
        auto out_8 = calculateNodeList["relu4"]->forward({out_7});

        // std::cout <<"can forward conv4" << std::endl;

        auto out_9 = calculateNodeList["flatten"]->forward({out_8});

        // std::cout <<"can flatten" << std::endl;
        // std::cout <<"flatten size is " << out_9->shape()[0] << " " << out_9->shape()[1] << std::endl;

        auto out_10 = calculateNodeList["linear1"]->forward({out_9});
        auto out_11 = calculateNodeList["relu5"]->forward({out_10});

        // std::cout <<"can forward linear 1" << std::endl;
        auto out_12 = calculateNodeList["linear2"]->forward({out_11});
        auto out_13 = calculateNodeList["relu6"]->forward({out_12});

        auto out_14 = calculateNodeList["linear3"]->forward({out_13});
        // std::cout <<"can forward linear 3" << out_14->getData() << std::endl<< std::endl;
        auto out_15 = calculateNodeList["sigmoid"]->forward({out_14});
        // std::cout <<"can forward !"<<std::endl;
        return out_15;
    }
};

class MLP:public Module<float>{
public:
    MLP(){
        calculateNodeList["flatten"] = std::make_shared<Flatten<float>>();
        calculateNodeList["linear1"] = std::make_shared<Linear<float>>(784 , 256);
        calculateNodeList["linear2"] = std::make_shared<Linear<float>>(256 , 256);
        calculateNodeList["linear3"] = std::make_shared<Linear<float>>(256 , 256);
        calculateNodeList["linear4"] = std::make_shared<Linear<float>>(256 , 64);
        calculateNodeList["linear5"] = std::make_shared<Linear<float>>(64 , 1);
        calculateNodeList["relu1"] = std::make_shared<Sigmoid<float>>();
        calculateNodeList["relu2"] = std::make_shared<Sigmoid<float>>();
        calculateNodeList["relu3"] = std::make_shared<Sigmoid<float>>();
        calculateNodeList["relu4"] = std::make_shared<Sigmoid<float>>();
        calculateNodeList["sigmoid"] = std::make_shared<Sigmoid<float>>();
    }
    
    std::shared_ptr<Tensor<float>> forward(std::initializer_list<std::shared_ptr<Tensor<float>>> data) override{
        auto input = *(data.begin());
        auto out1 = calculateNodeList["flatten"]->forward({input});

        auto out2 = calculateNodeList["linear1"]->forward({out1});
        auto out3 = calculateNodeList["relu1"]->forward({out2});

        auto out4 = calculateNodeList["linear2"]->forward({out3});
        auto out5 = calculateNodeList["relu2"]->forward({out4});

        auto out6 = calculateNodeList["linear3"]->forward({out5});
        auto out7 = calculateNodeList["relu3"]->forward({out6});

        auto out8 = calculateNodeList["linear4"]->forward({out7});
        auto out9 = calculateNodeList["relu4"]->forward({out8});

        auto out10 = calculateNodeList["linear5"]->forward({out9});
        auto out11 = calculateNodeList["sigmoid"]->forward({out10});
        return out11;
    }
};


int main(){

    // auto data = std::make_shared<Tensor<int>>(std::initializer_list<int>({4 , 1 , 4,4}));
    // data->getData() <<0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,
    // 37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63; 
    // std::cout << "the data is " <<std::endl << data->getData() << std::endl <<std::endl;
    // auto conv = std::make_shared<Conv2d<int>>(1,2,3,1);
    // conv->preTensorNodes["weights"]->getData().setOnes();
    // std::cout << "the weights is " <<std::endl << conv->preTensorNodes["weights"]->getData() << std::endl <<std::endl;
    // auto output = conv->forward({data});
    // std::cout << "the img2col is " <<std::endl << conv->m_img2colData->getData() << std::endl <<std::endl;
    // std::cout << "the output is " <<std::endl << output->getData() << std::endl <<std::endl;
    // output->getGrad().setOnes();
    // std::cout << "the output grad is " <<std::endl << output->getGrad() << std::endl <<std::endl;
    // output->backward();
    // std::cout << "the weights grad is " <<std::endl << conv->preTensorNodes["weights"]->getGrad() << std::endl <<std::endl;
    // std::cout << "the data grad is " <<std::endl << data->getGrad() << std::endl <<std::endl;


    std::string dirName = "D:\\pytorch_dataset\\mnist\\small_train";
    auto dataset = std::make_shared<MnistDataset<float>>(dirName);
    auto dataLoader = std::make_shared<DataLoader<float>>(dataset , 10 , 4 , true);
    auto net = std::make_shared<LeNet>();
    // auto net = std::make_shared<MLP>();
    auto opti = std::make_shared<SGD<float>>(net , 0.001f);
    std::cout << "the learn num is " << opti->learnParameterList.size() <<  std::endl;
    auto bceLoss = std::make_shared<BCELoss<float>>( );
    for(int i = 0 ; i < 5 ; ++i){
        dataLoader->reset();
        while(!dataLoader->empty()){
            std::cout << "The epoch is " << i << std::endl;
            auto dataAndLabel = dataLoader->getData();
            auto data = dataAndLabel.first;
            // data->getData().setRandom();
            auto label = dataAndLabel.second;
            // std::cout << "the data is " <<std::endl << data->getData().transpose()<< std::endl <<std::endl;
            std::cout << "the label is " <<std::endl << label->getData() << std::endl <<std::endl;
            auto output = net->forward({data});
            std::cout << "the output is " <<std::endl << output->getData() << std::endl <<std::endl;
            auto loss = bceLoss->forward({output , label});

            std::cout << "the loss is " <<std::endl << loss->getData() << std::endl <<std::endl;
            loss->backward();
            opti->step();
            opti->zeroGrad();
        }
    }

    return 0;
}
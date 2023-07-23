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
#include"DatasetBase.cu"
#include"DataLoader.cu"
#include<unsupported\Eigen\CXX11\Tensor>

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
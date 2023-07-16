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
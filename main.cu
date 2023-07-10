#include<iostream>
#include<Eigen\Dense>
#include"Tensor.cu"
#include<string>
#include<exception>
#include"BasicCalculateNode.cu"
#include<memory>
#include"ActivationFunction.cu"

int main(){

    auto data = std::make_shared<Tensor<float>>(std::initializer_list<int>({5,5}));
    data->getData() = Eigen::ArrayXXf::Constant(5 , 5 , 1);
    std::cout << "the data is " << std::endl << data->getData() << std::endl;
    Tanh<float> ac;
    auto output = ac.forward({data});
    output->getGrad() = Eigen::ArrayXXf::Constant(5,5,0.2);
    std::cout << "the grad is " << std::endl << output->getGrad() << std::endl;
    output->backward();
    std::cout << "the data is " << std::endl << output->getData() << std::endl;
    std::cout << "the grad is " << std::endl << data->getGrad() << std::endl;




    // Linear<int> mlp(2,3 );
    // auto x = std::make_shared<Tensor<int>>(std::initializer_list<int>({3,2}));
    // auto output = mlp.forward({x});

    // std::cout << "The input data is " << std::endl << x->getData() << std::endl;
    // std::cout << "The weights data is " << std::endl << mlp.preTensorNodes["weights"]->getData() << std::endl;
    // std::cout << "The bias data is " << std::endl << mlp.preTensorNodes["bias"]->getData() << std::endl;
    // std::cout << "The output data is " << std::endl << output->getData() << std::endl;

    // output->getGrad() = Eigen::ArrayXXi::Random(3,3);
    // std::cout << "The output grad is " << std::endl << output->getGrad() << std::endl << std::endl ;
    // output->backward();
    // std::cout << "The input grad is " << std::endl << x->getGrad() << std::endl << std::endl;
    // std::cout << "The weights grad is " << std::endl << mlp.preTensorNodes["weights"]->getGrad() << std::endl<< std::endl;
    // std::cout << "The bias grad is " << std::endl << mlp.preTensorNodes["bias"]->getGrad() << std::endl << std::endl;


    // std::cout <<"the after grad is" << std::endl<<  mydata.getGrad() <<std::endl;
    return 0;
}
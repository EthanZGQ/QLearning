#ifndef ELEMENT_WISE
#define ELEMENT_WISE

#include"cuda.h"
#include"cuda_runtime.h"

#define BLOCKSIZE 256


template<typename FUNC , typename O , typename... N_INPUT>
__global__ void elementComputeKernal(FUNC func , size_t Len , O * output , N_INPUT* ... numInput){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= Len) return;
    output[tid] = func(numInput[tid]...);
}

template<typename FUNC , typename O ,typename... N_INPUT >
void prepareAndCompute(FUNC func , size_t Len , O * output , N_INPUT * ... numInput){
    int blockNum = (Len + BLOCKSIZE - 1) / BLOCKSIZE;
    elementComputeKernal<<<blockNum , BLOCKSIZE>>>(func , Len , output , numInput...);
}

template<typename FUNC , typename O , typename A>
void Unary(FUNC func , size_t Len ,O* output , A * input_1){
    prepareAndCompute(func , Len , output , input_1);
}

template<typename FUNC , typename O , typename A , typename B>
void Binary(FUNC func , size_t Len ,O* output , A * input_1 , B * input_2){
    prepareAndCompute(func , Len , output , input_1 , input_2);
}

template<typename FUNC , typename O , typename A , typename B , typename C>
void Ternary(FUNC func , size_t Len ,O* output , A * input_1 , B * input_2 , C * input_3){
    elemetCompute(func , Len , output , input_1 , input_2 , input_3);
}

#endif // ELEMENT_WISE
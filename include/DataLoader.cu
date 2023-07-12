#ifndef DATA_LOADER
#define DATA_LOADER

#include<memory>
#include<deque>
#include<random>
#include"Tensor.cu"
#include"DatasetBase.cu"
#include<thread>
#include<condition_variable>
#include<vector>
#include<mutex>
#include<queue>
#include<utility>
#include<chrono>

template<class T>
class DataLoader{
private:
    std::shared_ptr<DatasetBase<T>> m_dataset;
    bool stop = false;
    bool m_shuffle;
    int m_batchSize;
    int m_numWorkers;
    int busyWorkers = 0;
    std::vector<std::thread> threadWorkers;
    std::condition_variable cv_consumer;
    std::condition_variable cv_producers;
    std::mutex mx;
    std::deque<std::pair<std::shared_ptr<Tensor<float>> , std::shared_ptr<Tensor<float>>>> dataPool;
    std::deque<int> idxList;

public:
    DataLoader(std::shared_ptr<DatasetBase<T>> dataset , int batchSize = 1 , int numWorkers = 0 , bool shuffle = true):
    m_dataset(dataset) , m_batchSize(batchSize) , m_numWorkers(numWorkers) , m_shuffle(shuffle){

    }

    ~DataLoader(){
        stop = true;
        cv_producers.notify_all();
        for(auto & tempThread : threadWorkers){
            if(tempThread.joinable()) tempThread.join();
        }
    }

    std::pair<std::shared_ptr<Tensor<T>> , std::shared_ptr<Tensor<T>>> getData(){
        std::unique_lock<std::mutex> lck(mx);
        while(dataPool.empty()){
            cv_consumer.wait(lck);
        }
        auto data = dataPool.front();
        dataPool.pop_front();
        if(idxList.size() >= m_numWorkers){
            cv_producers.notify_one();
        }
        return data;
    }

    bool empty(){
        return idxList.size() < m_numWorkers && busyWorkers == 0 && dataPool.empty();
    }

    void reset(){
        std::unique_lock<std::mutex> lck(mx);
        idxList.clear();
        for(int i = 0 ; i < m_dataset->len() ; ++i){
            idxList.push_back(i);
        }
        if(m_shuffle){
            std::srand(std::chrono::system_clock::now().time_since_epoch().count());
            std::random_shuffle(idxList.begin() , idxList.end());
        }
        dataPool.clear()
        cv_producers.notify_all();
    }

    void worker(){
        std::vector<int> threadIdxList;
        while(true){
            std::unique_lock<std::mutex> lck(mx);
            while(idxList.size() < m_numWorkers && || dataPool.size() + busyWorkers >= m_numWorkers){
                if(stop) return;
                cv_producers.wait(lck);
            }
            if(stop) return;
            ++busyWorkers;
            for(int i = 0 ; i < m_batchSize ; ++i){
                threadIdxList.push_back(idxList.front());
                idxList.pop_front();
            }
            lck.unlock();
            // xxxxx
            auto inputData = std::make_shared<Tensor<T>>({1});
            auto inputLabel = std::make_shared<Tensor<T>>({1});
            //xxx get the data
            lck.lock();
            --busyWorkers;
            dataPool.push_back({inputData , inputLabel});
            cv_consumer.notify_one();
        }

    }
};


#endif
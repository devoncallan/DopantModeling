#include <vector>
#include <unordered_map>
#include <iostream>
#include <random>
#include <chrono>

std::random_device rd;
std::mt19937 rng(rd());
std::uniform_real_distribution<double> dis(0.0, 1.0);

class PolymerMap
{
    private:
    std::vector<uint32_t> indexVector;
    std::unordered_map<uint32_t, uint32_t> indexMap;
    

    public:

    PolymerMap(uint32_t numPolymers) 
    {
        indexVector.reserve(numPolymers);
        indexMap.reserve(numPolymers);
        std::cout << indexVector.max_size() << std::endl;
        std::cout << indexMap.max_size() << std::endl;
        // std::cout << indexVector.capacity() << std::endl;
    }

    bool insert(const uint32_t &polymerIndex)
    {
        if(indexMap.count(polymerIndex)) return false;
        indexMap[polymerIndex] = indexVector.size();
        indexVector.push_back(polymerIndex);
        return true;
    }

    bool remove(const uint32_t &polymerIndex)
    {
        if (!indexMap.count(polymerIndex)) return false;
        int oldIndex = indexMap[polymerIndex];
        indexMap[indexVector[oldIndex] = indexVector.back()] = oldIndex;
        indexVector.pop_back();
        indexMap.erase(polymerIndex);
        return true;
    }

    bool insertOrRemoveIf(const uint32_t &polymerIndex, bool valid)
    {
        if (valid)
            return insert(polymerIndex);
        else
            return remove(polymerIndex);
    }

    uint32_t getRandomIndex()
    {
        return indexVector[dis(rng) * indexVector.size()];
    }

    int size()
    {
        return indexVector.size();
    }

    void printMap()
    {
        for (int i = 0; i < indexVector.size(); i++)
        {
            std::cout << indexVector[i] << " ";
        }
        std::cout << std::endl;
    }
};

using namespace std::chrono;

int main()
{
    std::vector<int> N_list = {1000, 10000, 100000, 1000000, 10000000, 100000000};
    
    for (int i = 0; i < N_list.size(); i++)
    {
        int N = N_list[i];
        PolymerMap polymerMap(N);
        
        
        high_resolution_clock::time_point begin;
        high_resolution_clock::time_point end;
        double t;
        
        std::cout << N << std::endl;
        std::cout << "INSERT : ";
        begin = high_resolution_clock::now();
        for (int i = 0; i < N; i++)
        {
            polymerMap.insert(i);
        }
        end = high_resolution_clock::now();
        t = duration_cast<nanoseconds>(end - begin).count();
        std::cout << t/1e9 << " s | " << t / N << "ns per event " << std::endl;
        
        std::cout << "RANDOM : ";
        begin = high_resolution_clock::now();
        for (int i = 0; i < N; i++)
        {
            polymerMap.getRandomIndex();
        }
        end = high_resolution_clock::now();
        t = duration_cast<nanoseconds>(end - begin).count();
        std::cout << t/1e9 << " s | " << t / N << "ns per event " << std::endl;
        
        std::cout << "REMOVE : ";
        begin = high_resolution_clock::now();
        for (int i = 0; i < N; i++)
        {
            polymerMap.remove(i);
        }
        end = high_resolution_clock::now();
        t = duration_cast<nanoseconds>(end - begin).count();
        std::cout << t/1e9 << " s | " << t / N << "ns per event " << std::endl;
        std::cout << std::endl;
    }
    return 0;
}

class DataStructure
{
    private:
    std::vector<int> indexVector;
    std::unordered_map<int, int> indexMap;

    public:

    DataStructure(int maxNumIndices) 
    {
        indexVector.reserve(maxNumIndices);
        indexMap.reserve(maxNumIndices);
    }

    bool insert(const int &index)
    {
        if(indexMap.count(index)) return false;
        indexMap[index] = indexVector.size();
        indexVector.push_back(index);
        return true;
    }

    bool remove(const int &index)
    {
        if (!indexMap.count(index)) return false;
        int oldIndex = indexMap[index];
        indexMap[indexVector[oldIndex] = indexVector.back()] = oldIndex;
        indexVector.pop_back();
        indexMap.erase(index);
        return true;
    }

    int getRandomIndex()
    {
        return indexVector[dis(rng) * indexVector.size()];
    }
};

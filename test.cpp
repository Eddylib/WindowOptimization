//
// Created by libaoyu on 18-11-5.
//
#include <iostream>
#include <eigen3/Eigen/Dense>
#include "src/backend/window_optimization.h"
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <bitset>
#include "utils.h"
#include "src/alg_utils.h"
class Accumulator11
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    float A;
    size_t num;

    inline void initialize()
    {
        A=0;
        memset(SSEData,0, sizeof(float)*4*1);
        memset(SSEData1k,0, sizeof(float)*4*1);
        memset(SSEData1m,0, sizeof(float)*4*1);
        num = numIn1 = numIn1k = numIn1m = 0;
    }

    inline void finish()
    {
        shiftUp(true);
        A=SSEData1m[0+0] + SSEData1m[0+1] + SSEData1m[0+2] + SSEData1m[0+3];
    }


    inline void updateSingle(
            const float val)
    {
        SSEData[0] += val;
        num++; numIn1++;
        shiftUp(false);
    }

    inline void updateSSE(
            const __m128 val)
    {
        _mm_store_ps(SSEData, _mm_add_ps(_mm_load_ps(SSEData),val));
        num+=4;
        numIn1++;
        shiftUp(false);
    }

    inline void updateSingleNoShift(
            const float val)
    {
        SSEData[0] += val;
        num++; numIn1++;
    }

    inline void updateSSENoShift(
            const __m128 val)
    {
        _mm_store_ps(SSEData, _mm_add_ps(_mm_load_ps(SSEData),val));
        num+=4;
        numIn1++;
    }



private:
    EIGEN_ALIGN16 float SSEData[4*1];
    EIGEN_ALIGN16 float SSEData1k[4*1];
    EIGEN_ALIGN16 float SSEData1m[4*1];
    float numIn1, numIn1k, numIn1m;


    void shiftUp(bool force)
    {
        if(numIn1 > 1000 || force)
        {
            _mm_store_ps(SSEData1k, _mm_add_ps(_mm_load_ps(SSEData),_mm_load_ps(SSEData1k)));
            numIn1k+=numIn1; numIn1=0;
            memset(SSEData,0, sizeof(float)*4*1);
        }

        if(numIn1k > 1000 || force)
        {
            _mm_store_ps(SSEData1m, _mm_add_ps(_mm_load_ps(SSEData1k),_mm_load_ps(SSEData1m)));
            numIn1m+=numIn1k;  numIn1k=0;
            memset(SSEData1k,0, sizeof(float)*4*1);
        }
    }
};
using namespace std;
int testacc(){
    Accumulator11 a;
    float update = 100000000;
    int times = 10000;
    a.initialize();
    timeval t[2];
    start(t);
    for (int i = 0; i < times ; ++i) {
        for (int j = 0; j < times; ++j) {
            a.updateSingleNoShift(update);
        }
        a.updateSingle(update);
    }
    stop(t);
    a.finish();
    cout<<"Accumulator result: "<<a.A<<" , cost: "<<duration(t)<<" ms"<<endl;
    float data=0;
    start(t);
    for (int i = 0; i < times ; ++i) {
        for (int j = 0; j < times; ++j) {
            data+=update;
        }
        data+=update;
    }
    stop(t);
    cout<<"direct add  result: "<<data<<" , cost: "<<duration(t)<<" ms"<<endl;
}
template <typename T,typename TB>
void print_2(T &&a,TB &&b){
    cout<<a<<", "<<b<<endl;
}
template <typename T>
void print_2(T && a){
    cout<<a<<endl;
}

template <typename T,typename T2, typename... Args>
void print_2(T &&a,T2 &&b,Args&&... rest){
    cout<<a<<", "<<b<<", "<<endl;
    cout<<sizeof...(Args)<<endl;
    cout<<sizeof...(rest)<<endl;
    print_2(std::forward<Args>(rest)...);
//    print_2(std::forward<Args>(rest)...);
    a=3;

}
int main(){
   Eigen::MatrixXf a = Eigen::MatrixXf();
    a.resize(3,3);
   a<<1,1,1,1,1,1,1,1,1;
    cout<<a.exp()<<endl;
}
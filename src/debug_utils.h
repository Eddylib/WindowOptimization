//
// Created by libaoyu on 18-11-7.
//

#ifndef WINDOWOPTIMIZATION_DEBUG_UTILS_H
#define WINDOWOPTIMIZATION_DEBUG_UTILS_H

#include "types.h"
class DataGenerator{
public:
    template <typename T>
    static T gen_data(){
        T ret;
        ret.setRandom();
        for (int i = 0; i < ret.rows(); ++i) {
            for (int j = 0; j < ret.cols(); ++j) {
                ret(i,j)+=0.5;
            }
        }
        return ret;
    }
    static Mat gen_cross_mat(const Mat &in){
        using namespace std;
        assert(in.rows() == 3);
        Mat ret;
        ret.resize(3,3);
        ret.setZero();
        ret(0,1) -= in(2);
        ret(0,2) = in(1);
        ret(1,2) -= in(0);

        ret(1,0) = in(2);
        ret(2,0) -= in(1);
        ret(2,1) = in(0);
        return ret;
    }
};

#endif //WINDOWOPTIMIZATION_DEBUG_UTILS_H

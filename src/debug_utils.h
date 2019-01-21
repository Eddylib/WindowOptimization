//
// Created by libaoyu on 18-11-7.
//

#ifndef WINDOWOPTIMIZATION_DEBUG_UTILS_H
#define WINDOWOPTIMIZATION_DEBUG_UTILS_H

#include <opencv2/core/hal/interface.h>
#include "alg_config.h"
#include "alg_utils.h"
#include <time.h>
#include <sys/time.h>
#define  dbcout cout<<__FILE__<<","<<__FUNCTION__<<","<<__LINE__<<": "

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

inline std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}
inline void start(timeval  *time){
    gettimeofday(time, NULL);
}
inline void stop(timeval  *time){
    gettimeofday(time+1, NULL);
}
inline long long duration(timeval  *time){
    return (time[1].tv_sec - time[0].tv_sec)*1000+(time[1].tv_usec - time[0].tv_usec)/1000;
}
#endif //WINDOWOPTIMIZATION_DEBUG_UTILS_H

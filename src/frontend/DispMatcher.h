//
// Created by libaoyu on 18-12-21.
//

#ifndef WINDOWOPTIMIZATION_DISPMATCHER_H
#define WINDOWOPTIMIZATION_DISPMATCHER_H


#include "../io/ImageLoader.h"
#include "../alg_config.h"
#include <opencv2/calib3d.hpp>

class DispMatcher {
    cv::Ptr<cv::StereoBM> stereoBM;
public:
    DispMatcher();
    virtual void do_disp_match(Frame &frame, cv::Mat &output);
};
class DispMatcherMy: public DispMatcher{
    int block_size = 10;
    float grad_thershold = _G_gradStereoThershold;
public:
    void do_disp_match(Frame &frame, cv::Mat &output) override;
    void set_block_size(int _size){block_size = _size;}
};

#endif //WINDOWOPTIMIZATION_DISPMATCHER_H

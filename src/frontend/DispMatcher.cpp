//
// Created by libaoyu on 18-12-21.
//

#include <opencv2/calib3d.hpp>
#include <Eigen/Core>
#include <opencv/cv.hpp>
#include <opencv2/core/eigen.hpp>

#include "DispMatcher.h"
using namespace cv;
using namespace std;
void DispMatcher::do_disp_match(Frame &frame, cv::Mat &output) {
    stereoBM->compute(frame.img_l,frame.img_r,output);
}

DispMatcher::DispMatcher() {
    stereoBM = StereoBM::create();
    auto ptr = stereoBM.get();
    ptr->setPreFilterSize(5);
//    ptr->setNumDisparities(160);
    ptr->setPreFilterCap(61);
    ptr->setTextureThreshold(507);
    ptr->setSpeckleRange(8);
    ptr->setSpeckleWindowSize(9);
}

void DispMatcherMy::do_disp_match(Frame &frame, cv::Mat &output) {
    return;
}

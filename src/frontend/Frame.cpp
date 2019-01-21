//
// Created by libaoyu on 19-1-9.
//

#include "Frame.h"
#include <opencv/cv.hpp>
#include <opencv2/core/eigen.hpp>
using namespace std;
Frame::Frame(Frame &&src)noexcept{
    this->img_l = std::move(src.img_l);
    this->img_r = std::move(src.img_r);
    this->time_step = src.time_step;
    cout<<__FUNCTION__<<" move construct"<<endl;
}

Frame::Frame(const Frame &src) {
    this->img_l = src.img_l;
    this->img_r = src.img_r;
    this->time_step = src.time_step;
    cout<<__FUNCTION__<<" copy construct"<<endl;
}

Frame &Frame::operator=(Frame &&src) noexcept {
    this->img_l = std::move(src.img_l);
    this->img_r = std::move(src.img_r);
    this->time_step = src.time_step;
    return *this;
}

Frame &Frame::operator=(const Frame &src) {
    this->img_l = src.img_l;
    this->img_r = src.img_r;
    this->time_step = src.time_step;
    return *this;
}

Frame::Frame(cv::Mat &&_l_img, cv::Mat &&_r_img, long long int _time_step, int id) :
        img_l(_l_img),img_r(_r_img),time_step(_time_step),id(id){

    // 计算梯度
    cv::Scharr(img_l,gradlx,CV_32F,1,0);
    cv::Scharr(img_l,gradly,CV_32F,0,1);
    cv::Scharr(img_r,gradrx,CV_32F,1,0);
    cv::Scharr(img_r,gradry,CV_32F,0,1);

    W = img_l.cols;
    H = img_l.rows;
    img_data_l = new Eigen::Vector4f[W*H];
//    cv::imshow("gradRxabs",gradRxAbs);
//    cv::imshow("gradRx",gradRxAbs>_G_gradStereoThershold);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            float color,dx,dy,nd;
            color = img_l.at<float>(i,j);
            dx = gradlx.at<float>(i,j);
            dy = gradly.at<float>(i,j);
            nd = sqrtf(dx*dx + dy*dy);
            img_data_l[i*W + j][0] =color;
            img_data_l[i*W + j][1] =dx;
            img_data_l[i*W + j][2] =dy;
            img_data_l[i*W + j][3] =nd;
            color = img_r.at<float>(i,j);
            dx = gradrx.at<float>(i,j);
            dy = gradry.at<float>(i,j);
            nd = sqrtf(dx*dx + dy*dy);
            img_data_r[i*W + j][0] =color;
            img_data_r[i*W + j][1] =dx;
            img_data_r[i*W + j][2] =dy;
            img_data_r[i*W + j][3] =nd;
        }
    }
}

Frame::~Frame() {
    delete[] img_data_l;
    delete[] img_data_r;
}

void Frame::selectPoints() {
    // 选择： pad领域内最大的点， pad随实际情况增大
}

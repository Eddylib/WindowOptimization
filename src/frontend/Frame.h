//
// Created by libaoyu on 19-1-9.
//

#ifndef WINDOWOPTIMIZATION_FRAME_H
#define WINDOWOPTIMIZATION_FRAME_H



#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <iostream>
class Frame{
public:
    cv::Mat img_l;
    cv::Mat img_r;
    cv::Mat gradlx;
    cv::Mat gradly;
    cv::Mat gradrx;
    cv::Mat gradry;
    Eigen::Vector4f *img_data_l;  // gray dx dy |d|
    Eigen::Vector4f *img_data_r;  // gray dx dy |d|
    int *pt_status_l;

    Eigen::Matrix4d bodyPose;
    int W;
    int H;
    int id = 0;
    Frame(cv::Mat &&_l_img,cv::Mat &&_r_img,long long int _time_step, int id);
    ~Frame();

    Frame(const Frame &src);
    Frame(Frame && src) noexcept;
    Frame &operator = (const Frame & src);
    Frame &operator = (Frame && src) noexcept;
    Frame() = default;
    long long int time_step = 0;
    // 选择特征点， 填充pt_status_l字段，只对左图选特征点，
    void selectPoints();
};

#endif //WINDOWOPTIMIZATION_FRAME_H

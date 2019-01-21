//
// Created by libaoyu on 18-12-21.
//

#ifndef WINDOWOPTIMIZATION_IMAGELOADER_H
#define WINDOWOPTIMIZATION_IMAGELOADER_H

#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <iostream>
#include "../frontend/Frame.h"
class Moveable{
public:
    Moveable(){}
    Moveable(Moveable &&src);
    Moveable(const Moveable &src);
    Moveable&operator=(const Moveable &src);
    Moveable&operator=(Moveable &&src);
};


class ImageLoader {
public:
    virtual int num_position() = 0;
    virtual Frame get_frame(int id) = 0;
    virtual ~ImageLoader() = default;
};
class CamInfo{
    std::string data_path;
public:
    Eigen::Matrix4d tToBody;
    int rate;
    int imgH;
    int imgW;
    Eigen::Matrix3d camK;
    std::string distortion_model;
    cv::Mat distortion_coefficients;
    std::vector<std::string> flist;
    std::vector<std::string> fFulllist;
    std::vector<long long> timestep;
    std::string prefix;
    CamInfo(const std::string &data_root,const std::string &cam_yaml_prefix);
    std::string sensor_type;

};
class MAVImageLoader:public ImageLoader{
public:
    std::string data_root;
    CamInfo leftCam;
    CamInfo rightCam;
    explicit MAVImageLoader(const std::string &mav_root);
    int num_position() override;
    Frame get_frame(int id) override;
};


#endif //WINDOWOPTIMIZATION_IMAGELOADER_H

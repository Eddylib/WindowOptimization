//
// Created by libaoyu on 18-12-21.
//

#include "ImageLoader.h"
#include "../alg_config.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <opencv/cv.hpp>
#include <opencv2/core/eigen.hpp>
#include "../alg_utils.h"
using namespace std;
int MAVImageLoader::num_position() {
    assert(leftCam.fFulllist.size() == rightCam.fFulllist.size());
    return static_cast<int>(leftCam.fFulllist.size());
}

Frame MAVImageLoader::get_frame(int idx) {
    cv::Mat grayL,grayR;
    cv::Mat udisL,udisR;
    cv::Mat gradLx,gradRx;
    cv::Mat gradLy,gradRy;
    cv::Mat leftCamK,rightCamK;
    cv::eigen2cv(leftCam.camK,leftCamK);
    cv::eigen2cv(rightCam.camK,rightCamK);
    cv::cvtColor(cv::imread(leftCam.fFulllist[idx]),grayL,CV_RGB2GRAY);
    cv::cvtColor(cv::imread(rightCam.fFulllist[idx]),grayR,CV_RGB2GRAY);
    cv::undistort(grayL,udisL,leftCamK,leftCam.distortion_coefficients);
    cv::undistort(grayR,udisR,rightCamK,leftCam.distortion_coefficients);
    return Frame(
            std::move(udisL),
            std::move(udisR),
            leftCam.timestep[idx],
            idx);
}

MAVImageLoader::MAVImageLoader(const std::string &mav_root):
data_root(mav_root),
leftCam(data_root,"cam0"),
rightCam(data_root,"cam1") {
    string cam_yaml = path_join({data_root,});

}

CamInfo::CamInfo(const std::string &data_root, const std::string &cam_yaml_prefix) {
    std::string yaml_file = path_join({data_root,cam_yaml_prefix,"sensor.yaml"});
    std::string csv_file = path_join({data_root,cam_yaml_prefix,"sensor.yaml"});
    data_path = path_join({data_root,cam_yaml_prefix,"data"});
    YAML::Node yaml_node = YAML::LoadFile(yaml_file);

    sensor_type = yaml_node["sensor_type"].as<string>();

    // 外参
    YAML::Node T_BS = yaml_node["T_BS"];
    YAML::Node TBSData = T_BS["data"];
    for(size_t i=0; i<TBSData.size();i ++){
//        cout<<TBSData[i].as<float>()<<endl;
        ((double*)tToBody.data())[i] = TBSData[i].as<float>();
    }
    tToBody.transposeInPlace();
//    cout<<tToBody<<endl;

    //内参
    imgW = yaml_node["resolution"][0].as<int>();
    imgH = yaml_node["resolution"][1].as<int>();
    camK.setZero();
    camK(0,0) = yaml_node["intrinsics"][0].as<double>();
    camK(1,1) = yaml_node["intrinsics"][1].as<double>();
    camK(0,2) = yaml_node["intrinsics"][2].as<double>();
    camK(1,2) = yaml_node["intrinsics"][3].as<double>();
    camK(2,2) = 1;
//    cout<<camK<<endl;
//    cout<<imgW<<" "<<imgH<<endl;

    //畸变参数
    int numCoeff = static_cast<int>(yaml_node["distortion_coefficients"].size());
    distortion_coefficients.create(1,numCoeff,CV_64F);
    for(int i=0; i<numCoeff; i++){
        distortion_coefficients.at<double>(i)= yaml_node["distortion_coefficients"][i].as<double>();
    }

    //文件列表与时间戳
    string data_list_file = path_join({data_root,cam_yaml_prefix,"data.csv"});
    fstream data_list_stream(data_list_file);
    string tmp;
    getline(data_list_stream,tmp);
    long long time;
    char dot;
    string filename;
    while(!data_list_stream.eof()){
        data_list_stream>>time;
        data_list_stream>>dot;
        data_list_stream>>filename;
//        cout<<time<<","<<filename<<endl;
        flist.push_back(filename);
        fFulllist.push_back(path_join({data_root,cam_yaml_prefix,"data",filename}));
//        cout<<fFulllist.back()<<endl;
        timestep.push_back(time);
        data_list_stream.get(); // \n...
        data_list_stream.get(); // \r...
        if(data_list_stream.peek() == EOF){
            break;
        }
    }
}



Moveable::Moveable(Moveable &&src) {
    cout<<__FUNCTION__<<" move construct"<<endl;
}

Moveable::Moveable(const Moveable &src) {
    cout<<__FUNCTION__<<" copy construct"<<endl;

}

Moveable &Moveable::operator=(const Moveable &src) {
    cout<<__FUNCTION__<<" copy set"<<endl;

}

Moveable &Moveable::operator=(Moveable &&src) {
    cout<<__FUNCTION__<<" move set"<<endl;
}

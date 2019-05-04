//
// Created by libaoyu on 19-1-3.
//

#ifndef WINDOWOPTIMIZATION_ALG_UTILS_H
#define WINDOWOPTIMIZATION_ALG_UTILS_H

#ifdef BUILD_FROUNT
#include <opencv2/core.hpp>
#endif
#include <eigen3/Eigen/Dense>
#include "alg_config.h"

typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vector;

//typedef Eigen::Matrix<SCALAR,FRAME_DIM,1> CamState;
//typedef Eigen::Matrix<SCALAR,RES_DIM,POINT_DIM> Jdrdp_t;
//typedef Eigen::Matrix<SCALAR,RES_DIM,FRAME_DIM> Jdrdxi_t;
//typedef Eigen::Matrix<SCALAR,POINT_DIM,POINT_DIM> Jdrdp2_t;
//typedef Eigen::Matrix<SCALAR,FRAME_DIM,FRAME_DIM> Jdrdxi2_t;
//typedef Eigen::Matrix<SCALAR,WINDOW_SIZE_MAX*FRAME_DIM,WINDOW_SIZE_MAX*FRAME_DIM> Hxixi_t;
//typedef Eigen::Matrix<SCALAR,FRAME_DIM,FRAME_DIM> Adjoint;
//template <typename T>
//void nms_(cv::Mat &grad);
std::string path_join(std::initializer_list<std::string> strlist);
float fixs16_2_float(int16_t in);

extern int staticPattern[8][2];


#endif //WINDOWOPTIMIZATION_ALG_UTILS_H

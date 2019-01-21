//
// Created by libaoyu on 18-11-7.
//

#ifndef WINDOWOPTIMIZATION_RESIDUAL_H
#define WINDOWOPTIMIZATION_RESIDUAL_H

#include <iostream>
#include "../debug_utils.h"
#include "../alg_config.h"
class Camera;
class Residual;
class Point;
class Residual{
public:
    Camera *host;
    Camera *target;
    Point *point;
    Residual(Camera *_host,Camera *_target,Point *_point,Eigen::Matrix<SCALAR,RES_DIM,1>);
    Residual(Camera *_host,Camera *_target,Point *_point);
    Jdrdp_t jdrdp;
    Jdrdxi_t jdrdxi_th; //相对位姿
    Adjoint getAdjH();
    Adjoint getAdjT();
    Jdrdxi_t getJthAdjH();
    Jdrdxi_t getJthAdjT();
    Eigen::Matrix<SCALAR,RES_DIM,1> resdata;
    Eigen::Matrix<SCALAR,FRAME_DIM,1> jthAdjH_r;   //  jdrdxi_th' * adjH *r 绝对位姿的导数乘以残差
    Eigen::Matrix<SCALAR,FRAME_DIM,1> jthAdjT_r;   //  jdrdxi_th' * adjT *r
    Eigen::Matrix<SCALAR,FRAME_DIM,1> jth_r;       //  jdrdxi_th' *r
    Eigen::Matrix<SCALAR,POINT_DIM,1> jp_r;        //  jdrdp' *r

};


#endif //WINDOWOPTIMIZATION_RESIDUAL_H

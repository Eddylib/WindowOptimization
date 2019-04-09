//
// Created by libaoyu on 18-11-7.
//
#pragma once

#include <iostream>
#include "../debug_utils.h"
#include "../alg_config.h"
#include "Camera.h"
#include "Point.h"
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
class Camera;

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
class Point;
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>

class Residual{
public:
    typedef Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> Camera_t;
    typedef Point<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> Point_t;
    typedef Eigen::Matrix<SCALAR,RES_DIM,POINT_DIM> Jdrdp_t;
    typedef Eigen::Matrix<SCALAR,RES_DIM,FRAME_DIM> Jdrdxi_t;
    typedef Eigen::Matrix<SCALAR,FRAME_DIM,FRAME_DIM> Adjoint;
    Camera_t *host;
    Camera_t *target;
    Point_t *point;
    Residual(Camera_t *_host,Camera_t *_target,Point_t *_point,Eigen::Matrix<SCALAR,RES_DIM,1>);
    Residual(Camera_t *_host,Camera_t *_target,Point_t *_point);
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
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Residual(Camera_t *_host, Camera_t *_target, Point_t *_point):
        Residual(_host,_target,_point,DataGenerator::gen_data<Eigen::Matrix<SCALAR,RES_DIM,1>>()){
    //DataGenerator::gen_data<Eigen::Matrix<SCALAR,RES_DIM,1>>()
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Residual(Camera_t *_host,Camera_t *_target,Point_t *_point,Eigen::Matrix<SCALAR,RES_DIM,1> _resdata):
        host(_host),target(_target),point(_point),
        jdrdp(DataGenerator::gen_data<Jdrdp_t>()),
        jdrdxi_th(DataGenerator::gen_data<Jdrdxi_t>()),
        resdata(_resdata){
//    assert(host && target && point);
//    *point->Eik[host->getid()] += getJthAdjH().transpose()*jdrdp;
//    *point->Eik[target->getid()] += getJthAdjT().transpose()*jdrdp;
//    point->getResiduals().push_back(this);
//    // one for target, one for host
//    point->C+= jdrdp.transpose()*jdrdp;
//    jthAdjH_r = getJthAdjH().transpose()*resdata;   //  jdrdxi_th * adjH *r 绝对位姿的导数乘以残差
//    jthAdjT_r = getJthAdjT().transpose()*resdata;   //  jdrdxi_th * adjT *r
//    jth_r = jdrdxi_th.transpose()*resdata;          //  jdrdxi_th *r
//    jp_r = jdrdp.transpose()*resdata;               //  jdrdp * r
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
typename Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Adjoint
Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::getAdjH() {
    return host->getAdjointAsH();
}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
typename Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Adjoint
Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::getAdjT() {
    return target->getAdjointAsT();
}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
typename Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Jdrdxi_t
Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::getJthAdjH() {
    return jdrdxi_th*getAdjH();
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
typename  Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Jdrdxi_t
Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::getJthAdjT() {
    return jdrdxi_th;/* *I */
}
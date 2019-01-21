//
// Created by libaoyu on 18-11-7.
//

#include "Residual.h"
#include "Point.h"
#include "Camera.h"
Residual::Residual(Camera *_host, Camera *_target, Point *_point):
Residual(_host,_target,_point,DataGenerator::gen_data<Eigen::Matrix<SCALAR,RES_DIM,1>>()){
    //DataGenerator::gen_data<Eigen::Matrix<SCALAR,RES_DIM,1>>()
}
Residual::Residual(Camera *_host,Camera *_target,Point *_point,Eigen::Matrix<SCALAR,RES_DIM,1> _resdata):
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
Adjoint Residual::getAdjH() {
    return host->getAdjointAsH();
}

Adjoint Residual::getAdjT() {
    return target->getAdjointAsT();
}

Jdrdxi_t Residual::getJthAdjH() {
    return jdrdxi_th*getAdjH();
}
Jdrdxi_t Residual::getJthAdjT() {
    return jdrdxi_th;/* *I */
}
//
// Created by libaoyu on 18-11-7.
//
#pragma once


#include <vector>
#include "../alg_config.h"
#include "../debug_utils.h"
#include "Residual.h"
#include "Point.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include "type_pre_declares.h"

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
class Camera{
public:
    typedef Point<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> Point_t;
    typedef ResidualBase<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> Residual_t;
    typedef  typename Residual_t::Adjoint Adjoint;
private:
    int id;
    std::vector<Point_t*> points;

public:
    using CamState = Eigen::Matrix<SCALAR,FRAME_DIM,1>;

    explicit Camera(int _id);
    explicit Camera(int _id, CamState camState);
    Point_t *newPoint();
    std::vector<Point_t*> &getPoints(){return points;}
    int getid(){ return id;}
    Vector delta;
    // einvCJpR = jdrdx*r - E*inv(C)*(Jdrdp * r)   (w=Jdrdp * r)
    // (CameraNum * FRAME_DIM) * 1
    //
    // use only when optimize,各个相机的right加起来构成了优化式子的右边
    Eigen::Matrix<SCALAR,FRAME_DIM,1> jx_r;
    Eigen::Matrix<SCALAR,FRAME_DIM,1> einvCJpR;
    //point被margin掉而导致的信息增值
    Eigen::Matrix<SCALAR,FRAME_DIM,1> einvCJpRMargedP;

    Eigen::Matrix<SCALAR,FRAME_DIM,1> prior_h;
    CamState state;
    Sophus::SE3<SCALAR> se3State;
    Adjoint getAdjointAsH();
    Adjoint getAdjointAsT();
};

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Camera(int _id):Camera(_id,DataGenerator::gen_data<CamState>()){

}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Camera(int _id, CamState camState):id(_id),state(camState){
    einvCJpR.setZero();
    jx_r.setZero();
    einvCJpRMargedP.setZero();
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
Point<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> *
Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::newPoint() {
    auto *ret = new Point<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>(static_cast<int>(points.size()));
    points.push_back(ret);
    return ret;
}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
typename Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Adjoint Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::getAdjointAsH() {
    Adjoint ret;
    ret.setZero();
    Mat R = (DataGenerator::gen_cross_mat(state.block(0,0,3,1))).exp();
    auto that = DataGenerator::gen_cross_mat(state.block(3,0,3,1));
    ret.block(0,0,3,3) = R;
    ret.block(0,3,3,3) = that*R;
    ret.block(3,3,3,3) = R;
    return ret;
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
typename Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Adjoint
Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::getAdjointAsT() {
    return Adjoint::Identity();
}
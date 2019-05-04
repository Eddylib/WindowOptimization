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
    using CamState = typename Eigen::Matrix<SCALAR,FRAME_DIM,1>;
private:
    int id;
    std::vector<Point_t*> points;
    CamState state;
    Sophus::SE3<SCALAR> se3State;
    void stateToSE3(){
            // camera[0,1,2] are the angle-axis rotation.
            // camera[3,4,5] are the translation.
            // 6 focal
            // 7,8 second and fourth order radial distortion.
            Eigen::Vector3d angleAxisData(state[0],state[1],state[2]);
            Eigen::Vector3d translation(state[3],state[4],state[5]);
            double radian = angleAxisData.norm();
            angleAxisData /= radian;
            Eigen::Quaterniond quaternion;
            quaternion = Eigen::AngleAxisd(radian,angleAxisData);
            se3State = Sophus::SE3<SCALAR>(quaternion,translation);
    }
    double *datahog;
public:

    explicit Camera(int _id);
    explicit Camera(int _id, double *data);
    Point_t *newPoint();
    const CamState &getState()const {return state;}
    void applyDelta(SCALAR lr,const Vector& delta){
        lr = lr;
        Eigen::Matrix<double,6,1> se3log;
        Sophus::SE3<double> se3_delta = Sophus::SE3<double>::exp(delta.block(0,0,6,1));

        const Sophus::SE3<SCALAR> &se3_x = se3State;
        Sophus::SE3<double> se3_x_plus_delta = se3_delta*se3_x;

        Eigen::AngleAxisd angleAxisd_x_plus_delta(se3_x_plus_delta.rotationMatrix());

        state.block(0,0,3,1) = angleAxisd_x_plus_delta.angle()*angleAxisd_x_plus_delta.axis();
        state.block(3,0,3,1) = se3_x_plus_delta.translation();

        state[6] += delta[6];
        state[7] += delta[7];
        state[8] += delta[8];

        if(datahog){
            Eigen::Map<CamState> publish(datahog); publish = state;
        }
        stateToSE3();
    }

    const Sophus::SE3<SCALAR>& getSE3State()const {return se3State;}
    void setSE3State(){

    }
    std::vector<Point_t*> &getPoints(){return points;}
    int getid(){ return id;}

    // einvCJpR = jdrdx*r - E*inv(C)*(Jdrdp * r)   (w=Jdrdp * r)
    // (CameraNum * FRAME_DIM) * 1
    //
    // use only when optimize,各个相机的right加起来构成了优化式子的右边
    Eigen::Matrix<SCALAR,FRAME_DIM,1> jx_r;
    Eigen::Matrix<SCALAR,FRAME_DIM,1> einvCJpR;
    //point被margin掉而导致的信息增值
    Eigen::Matrix<SCALAR,FRAME_DIM,1> einvCJpRMargedP;

    void clearStepInfo();
    Adjoint getAdjointAsH();
    Adjoint getAdjointAsT();
};
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::clearStepInfo(){
    einvCJpR.setZero();
    jx_r.setZero();
    einvCJpRMargedP.setZero();
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Camera(int _id):Camera(_id,DataGenerator::gen_data<CamState>(),nullptr){

}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Camera(int _id, double *data):id(_id),datahog(data){
    if(data){
        state = CamState(data);
    }
    einvCJpR.setZero();
    jx_r.setZero();
    einvCJpRMargedP.setZero();
    stateToSE3();
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
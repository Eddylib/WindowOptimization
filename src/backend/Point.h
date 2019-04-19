//
// Created by libaoyu on 18-11-7.
//
#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>
#include "../alg_config.h"
#include "../alg_utils.h"
#include "Camera.h"
#include "Residual.h"
#include "type_pre_declares.h"

typedef enum StateE{
    ACTIVED,IMMUENT,MARGINALIZED,TOBE_MARGINLIZE
}Status;
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
class Point{
public:
    int id{};
    typedef Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> Camera_t;
    using Residual_t = ResidualBase<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>;
//    Camera_t *host;
    using PointState = Eigen::Matrix<SCALAR,POINT_DIM,1>;
    Point(int _id);
    Point(int _id,PointState state);
    std::vector<Residual_t*> residuals;
    ~Point();
    int getId(){return id;}
    std::vector<Residual_t*>& getResiduals(){return residuals;}
    int hasResidualWithTarget(int camId);
    // i 为相机个数， k为点数，而k即指自己，所以为一维数组，8个，为窗口大小
    const Eigen::Matrix<SCALAR,FRAME_DIM,POINT_DIM> &getEik(int i)const{
        if(Eik[i] == nullptr){
            return zero;
        }else{
            return *Eik[i];
        }
    }
//    void setEik(int i, const Mat& data){
//        Eik[i] = data;
//    }
    void addEik(int i, const Eigen::Matrix<SCALAR,FRAME_DIM,POINT_DIM> & data){
        if(Eik[i] == nullptr){
            Eik[i] = new Eigen::Matrix<SCALAR,FRAME_DIM,POINT_DIM>(zero);
        }
        *Eik[i] += data;
    }
    const Eigen::Matrix<SCALAR,POINT_DIM,POINT_DIM> &getC()const{
        return C;
    }
    void addC(const Eigen::Matrix<SCALAR,POINT_DIM,POINT_DIM> &data){
        C += data;
    }
    Status getStatus()const{return status;}
    void setStatus(Status argin){status = argin;}
    const PointState &getJp_r()const{
        return jp_r;
    }
    void addJp_r(const PointState &other){
        jp_r += other;
    }
    const static Eigen::Matrix<SCALAR,FRAME_DIM,POINT_DIM> zero;
private:
    std::vector<Eigen::Matrix<SCALAR,FRAME_DIM,POINT_DIM> *> Eik;
    Eigen::Matrix<SCALAR,POINT_DIM,POINT_DIM> C;
    PointState jp_r;
    PointState prior;
    PointState pointState;
    Vector delta;
    Status status;

////前端数据结构
//    float idepth{};
//    float u{};
//    float v{};
//    //仅仅在视差匹配时使用
//    float u_stereo{};
//    float v_stereo{};
};

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
const Eigen::Matrix<SCALAR,FRAME_DIM,POINT_DIM>
        Point<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::zero;

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
Point<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::~Point(){
    for (int i = 0; i < residuals.size(); ++i) {
        delete residuals[i];
    }
    for (int i = 0; i < WINDOW_SIZE_MAX; ++i) {
        delete Eik[i];
    }
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
int Point<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::hasResidualWithTarget(int camId) {
//    int ret = 0;
//    for (int i = 0; i < residuals.size(); ++i) {
////        if(residuals[i]->getTarget()->getid() == camId){
//        if(residuals[i]->getTarget()->getid() == camId){
//            ret = 1;
//            break;
//        }
//    }
    return Eik[camId] != nullptr;
}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
Point<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Point(int _id):Point(_id, PointState()){
}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
Point<RES_DIM, FRAME_DIM, WINDOW_SIZE_MAX, POINT_DIM, SCALAR>::Point(
        int _id,
        PointState state):id(_id),prior(state),pointState(state){
//    for (int i = 0; i < WINDOW_SIZE_MAX; ++i) {
////        auto *tmp = new Eigen::Matrix<SCALAR,FRAME_DIM,POINT_DIM>();
////        tmp->setZero();
//        Eik.push_back(nullptr);
//    }
    Eik.resize(WINDOW_SIZE_MAX, nullptr);
    C.setZero();
    jp_r.setZero();
    status=ACTIVED;
}

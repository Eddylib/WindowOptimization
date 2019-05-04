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
    double *datahog;
public:
    int id{};
    typedef Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> Camera_t;
    using Residual_t = ResidualBase<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>;
//    Camera_t *host;
    using PointState = Eigen::Matrix<SCALAR,POINT_DIM,1>;
    Point(int _id);
    Point(int _id, double *data);
    std::vector<Residual_t*> residuals;
    ~Point();
    int getId(){return id;}
    std::vector<Residual_t*>& getResiduals(){return residuals;}
    int hasResidualWithTarget(int camId);
    bool ifHasEik(int i){return Eik[i];}
    // i 为相机个数， k为点数，而k即指自己，所以为一维数组，8个，为窗口大小
    const Eigen::Matrix<SCALAR,FRAME_DIM,POINT_DIM> &getEik(int i)const{
        if(Eik[i] == nullptr){
            return zero;
        }else{
            return *Eik[i];
        }
    }
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
    const PointState &getPointState()const{return pointState;}
    void getDelta(const Mat &camDelta){
        Eigen::Matrix<SCALAR,POINT_DIM,1> EikCamDelta;
        int camNum = camDelta.rows()/FRAME_DIM;
        for (int i = 0; i < camNum; ++i) {
            if(Eik[i]){
                EikCamDelta+=Eik[i]->transpose()*camDelta.block(i*FRAME_DIM,0,FRAME_DIM,1);
            }
        }
        delta = C.inverse()*(-jp_r-EikCamDelta);
    }
    void applyDelta(double lr){
        pointState += lr*delta ;
        if(datahog){
            Eigen::Map<PointState> publish(datahog);publish = pointState;
        }
    }
    void clearStepInfo(){
        C.setIdentity();
        C*=99999;
        for (int i = 0; i < Eik.size(); ++i) {
            if(Eik[i]){
                delete(Eik[i]);
            }
            Eik[i] = nullptr;
        }
        jp_r.setZero();
    }
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
        Eik[i] = nullptr;
    }
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
int Point<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::hasResidualWithTarget(int camId) {
    return Eik[camId] != nullptr;
}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
Point<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Point(int _id):Point(_id,nullptr){
}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
Point<RES_DIM, FRAME_DIM, WINDOW_SIZE_MAX, POINT_DIM, SCALAR>::Point(
        int _id, double *data):id(_id),datahog(data){
//    for (int i = 0; i < WINDOW_SIZE_MAX; ++i) {
////        auto *tmp = new Eigen::Matrix<SCALAR,FRAME_DIM,POINT_DIM>();
////        tmp->setZero();

//        Eik.push_back(nullptr);
//    }
    if(data){
        pointState = PointState(data);
        prior = pointState;
    }
    Eik.resize(WINDOW_SIZE_MAX, nullptr);
    C.setIdentity();
    C*=99999;
    jp_r.setZero();
    status=ACTIVED;
}

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
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
class Camera;

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
class Residual;

typedef enum StateE{
    ACTIVED,IMMUENT,MARGINALIZED,TOBE_MARGINLIZE
}State;
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
class Point{
public:
    int id;
    typedef Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> Camera_t;
    typedef Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> Residual_t;
    Camera_t *host;
    std::vector<Residual_t*> residuals;
    Point(int _id,Camera_t *_host):id(_id),host(_host){
        for (int i = 0; i < WINDOW_SIZE_MAX; ++i) {
            auto *tmp = new Eigen::Matrix<SCALAR,FRAME_DIM,POINT_DIM>();
            tmp->setZero();
            Eik.push_back(tmp);
            C.setZero();
        }
        jp_r.setZero();
        state=ACTIVED;
    }
    ~Point();
    int getId(){return id;}
    std::vector<Residual_t*>& getResiduals(){return residuals;}
    int hasResidualWithTarget(int camId);
    // i 为相机个数， k为点数，而k即指自己，所以为一维数组，8个，为窗口大小
    std::vector<Eigen::Matrix<SCALAR,FRAME_DIM,POINT_DIM> *> Eik;
    Eigen::Matrix<SCALAR,POINT_DIM,POINT_DIM> C;
    Vector pointV;  //Jdrdp * r (w)
    Eigen::Matrix<SCALAR,POINT_DIM,1> jp_r;
    Eigen::Matrix<SCALAR,POINT_DIM,1> prior;
    Vector delta;
    State state;

//前端数据结构
    float idepth;
    float u;
    float v;
    //仅仅在视差匹配时使用
    float u_stereo;
    float v_stereo;
};
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
Point<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::~Point(){
    for (int i = 0; i < residuals.size(); ++i) {
        delete residuals[i];
    }
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
int Point<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::hasResidualWithTarget(int camId) {
    int ret = 0;
    for (int i = 0; i < residuals.size(); ++i) {
        if(residuals[i]->target->getid() == camId){
            ret = 1;
            break;
        }
    }
    return ret;
}
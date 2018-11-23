//
// Created by libaoyu on 18-11-7.
//

#ifndef WINDOWOPTIMIZATION_CAMERA_H
#define WINDOWOPTIMIZATION_CAMERA_H


#include <vector>
#include "types.h"
#include "debug_utils.h"
class Camera;
class Residual;
class Point;
class Camera{
    int id;
    std::vector<Point*> points;

public:
    Camera(int _id):id(_id),state(DataGenerator::gen_data<CamState>()){}
    Point *newPoint();
    std::vector<Point*> &getPoints(){return points;}
    int getid(){ return id;}
    Vector delta;
    // einvCJpR = jdrdx*r - E*inv(C)*(Jdrdp * r)   (w=Jdrdp * r)
    // (CameraNum * FRAME_DIM) * 1
    //
    // use only when optimize,各个相机的right加起来构成了优化式子的右边
    Eigen::Matrix<SCALAR,FRAME_DIM,1> jx_r;
    Eigen::Matrix<SCALAR,FRAME_DIM,1> einvCJpR;

    Eigen::Matrix<SCALAR,FRAME_DIM,1> prior_h;
    CamState state;
    Adjoint getAdjointAsH();
    Adjoint getAdjointAsT();

};


#endif //WINDOWOPTIMIZATION_CAMERA_H

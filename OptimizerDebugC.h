//
// Created by libaoyu on 19-1-21.
//

#ifndef WINDOWOPTIMIZATION_OPTIMIZERDEBUGC_H
#define WINDOWOPTIMIZATION_OPTIMIZERDEBUGC_H

#include "src/backend/window_optimization.h"

class OptimizerDebugC: public WindowOptimizor<3,8,8,1,double>{
public:
    Mat generateCamUpdate();
    Mat generateTotalH();
    Mat generateTotalV();
    Mat generateTotalJ();
    Mat generateTotalR();
    Mat generateVCam();
    Mat generateVPoint();
    void reset();
    const static int RES_DIM = 3;
    const static int FRAME_DIM = 8;
    const static int WINDOW_SIZE_MAX = 8;
    const static int POINT_DIM = 1;
    typedef double SCALAR;
    typedef WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> Base_t;
    using Base_t::Camera_t;
    using Base_t::Point_t;
    using Base_t::Residual_t;
    void init(int camNum, int resNum);
    void newResidual(Camera_t *host, Camera_t *target, float possibilityCreateNewPoint = 0.01);
    const Mat get_B(){return B.block(0,0,cameras.size()*FRAME_DIM,cameras.size()*FRAME_DIM);}
    vector<Point_t*> getAllPoints();
    vector<Residual_t*> getAllResiduals();
    Mat genBUpdate();
    Mat genE_CInv_VPoint(const Mat &vPoint);
    void printInfo();
    void fetchInfo(int *camNum = nullptr,int *pointNum = nullptr, int *resNum = nullptr);
};
void test_init(OptimizerDebugC &system);

void test_add_camera(OptimizerDebugC &system);

void test_marg_point(OptimizerDebugC &system, float possibility);

#endif //WINDOWOPTIMIZATION_OPTIMIZERDEBUGC_H

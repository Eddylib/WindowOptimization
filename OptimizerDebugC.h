//
// Created by libaoyu on 19-1-21.
//

#ifndef WINDOWOPTIMIZATION_OPTIMIZERDEBUGC_H
#define WINDOWOPTIMIZATION_OPTIMIZERDEBUGC_H

#include "src/backend/window_optimization.h"
class OptimizerDebugC: public WindowOptimizor{
public:
    Mat generateCamUpdate();
    Mat generateTotalH();
    Mat generateTotalV();
    Mat generateTotalJ();
    Mat generateTotalR();
    Mat generateVCam();
    Mat generateVPoint();
    void reset();
    void init(int camNum, int resNum);
    void newResidual(Camera *host, Camera *target, float possibilityCreateNewPoint = 0.01);
    const Mat get_B(){return B.block(0,0,cameras.size()*FRAME_DIM,cameras.size()*FRAME_DIM);}
    vector<Point*> getAllPoints();
    vector<Residual*> getAllResiduals();
    Mat genBUpdate();
    Mat genE_CInv_VPoint(const Mat &vPoint);
    void printInfo();
    void fetchInfo(int *camNum = nullptr,int *pointNum = nullptr, int *resNum = nullptr);
};
void test_init(OptimizerDebugC &system);

void test_add_camera(OptimizerDebugC &system);

void test_marg_point(OptimizerDebugC &system, float possibility);

#endif //WINDOWOPTIMIZATION_OPTIMIZERDEBUGC_H

//
// Created by libaoyu on 19-4-18.
//

#pragma once
#include "type_pre_declares.h"


template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
class HessionBase{
public:
    using ResidualBase_t =  ResidualBase<RES_DIM, FRAME_DIM, WINDOW_SIZE_MAX, POINT_DIM, SCALAR>;
    using Point_t =  Point<RES_DIM, FRAME_DIM, WINDOW_SIZE_MAX, POINT_DIM, SCALAR>;
    using Camera_t =  Camera<RES_DIM, FRAME_DIM, WINDOW_SIZE_MAX, POINT_DIM, SCALAR>;
    // 当一个残差项进入系统时该做什么，主要是将信息添加到B中，（H的左上角）
    virtual void initapplyRes(ResidualBase_t *residual) = 0;
    // 清空marginlize相关数据结构
    virtual void clearMargedInfo() = 0;
    // 边缘化掉一个点,主要涉及到对marginlize相关数据结构的更新
    virtual void margOnePoint(Point_t *point, int camSize) = 0;
    // 使用舒尔布进行计算对B的更新量
    virtual void stepApplyPoint(Point_t *point, int camSize) = 0;
    // 进行一次迭代
    virtual void solveStep(Mat &right, Mat &delta) = 0;
    // 每次迭代前调用，清除stepApplyPoint相关数据内容
    virtual void resetForStep() = 0;
};

//
// Created by libaoyu on 18-11-7.
//
#pragma once

#include <iostream>
#include "../debug_utils.h"
#include "../alg_config.h"
#include "Camera.h"
#include "Point.h"
#include "type_pre_declares.h"
//实际算法应该实现特定的res
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
class ResidualBase{
public:
    using Camera_t = Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> ;
    using Point_t =  Point<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> ;
    using HessionBase_t =  HessionBase<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> ;
    using WindowOptimizor_t =  WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> ;
    typedef Eigen::Matrix<SCALAR,RES_DIM,POINT_DIM> Jdrdp_t;
    typedef Eigen::Matrix<SCALAR,RES_DIM,FRAME_DIM> Jdrdxi_t;
    typedef Eigen::Matrix<SCALAR,FRAME_DIM,FRAME_DIM> Adjoint;
    using VectorFrameDim = Eigen::Matrix<SCALAR,FRAME_DIM,1>;
    using VectorPointDim = Eigen::Matrix<SCALAR,POINT_DIM,1>;
    using ResData = Eigen::Matrix<SCALAR,RES_DIM,1>;
    virtual void initApplyData(HessionBase_t *hessionBase) = 0;
    virtual void computeRes() = 0;
    virtual void computeJ() = 0;

};

//
// Created by libaoyu on 19-4-19.
//

#ifndef WINDOWOPTIMIZATION_TYPES_H
#define WINDOWOPTIMIZATION_TYPES_H

#include "../src/backend/window_optimization.h"
const static int res_dim = 2;   // dertx, derty
const static int frame_dim = 9; // se3 x 6, f, p1, p2
const static int window_size = 60;
const static int point_dim = 3; // point in 3d space
#define Scalar double
using WindowOptimizor_t = WindowOptimizor<res_dim,frame_dim,window_size,point_dim, Scalar>;
using Camera_t = WindowOptimizor_t::Camera_t;
using Point_t = WindowOptimizor_t::Point_t ;
using Residual_t = WindowOptimizor_t::Residual_t ;
using HessionStruct_t = WindowOptimizor_t::HessionStruct_t;
#endif //WINDOWOPTIMIZATION_TYPES_H
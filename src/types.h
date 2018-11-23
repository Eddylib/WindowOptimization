//
// Created by libaoyu on 18-11-7.
//

#ifndef WINDOWOPTIMIZATION_TYPES_H
#define WINDOWOPTIMIZATION_TYPES_H

#define RES_DIM 3
#define FRAME_DIM 8
#define WINDOW_SIZE_MAX 8
#define POINT_DIM 1
#define SCALAR double

#include <eigen3/Eigen/Dense>
typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vector;

typedef Eigen::Matrix<SCALAR,FRAME_DIM,1> CamState;
typedef Eigen::Matrix<SCALAR,RES_DIM,POINT_DIM> Jdrdp_t;
typedef Eigen::Matrix<SCALAR,RES_DIM,FRAME_DIM> Jdrdxi_t;
typedef Eigen::Matrix<SCALAR,POINT_DIM,POINT_DIM> Jdrdp2_t;
typedef Eigen::Matrix<SCALAR,FRAME_DIM,FRAME_DIM> Jdrdxi2_t;
typedef Eigen::Matrix<SCALAR,WINDOW_SIZE_MAX*FRAME_DIM,WINDOW_SIZE_MAX*FRAME_DIM> Hxixi;
typedef Eigen::Matrix<SCALAR,FRAME_DIM,FRAME_DIM> Adjoint;
#endif //WINDOWOPTIMIZATION_TYPES_H

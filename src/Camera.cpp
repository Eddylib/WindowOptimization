//
// Created by libaoyu on 18-11-7.
//

#include "Camera.h"
#include "Point.h"
#include "Residual.h"
#include <eigen3/unsupported/Eigen/MatrixFunctions>
Point *Camera::newPoint() {
    Point *ret = new Point(static_cast<int>(points.size()), this);
    points.push_back(ret);
    return ret;
}

Adjoint Camera::getAdjointAsH() {
    Adjoint ret;
    ret.setZero();
    Mat R = (DataGenerator::gen_cross_mat(state.block(0,0,3,1))).exp();

    auto that = DataGenerator::gen_cross_mat(state.block(3,0,3,1));
    ret.block(0,0,3,3) = R;
    ret.block(0,3,3,3) = that*R;
    ret.block(3,3,3,3) = R;
    return ret;
}
Adjoint Camera::getAdjointAsT() {
    return Adjoint::Identity();
}
//
// Created by libaoyu on 19-4-27.
//

#ifndef CERES_LEARNING_UTILS_H
#define CERES_LEARNING_UTILS_H

#include <vector>
#include "baproblem.h"
void convert_cam_to_publis_format(double *state, double *published);

#ifdef ROS_DRAW
void draw_data(BAPublisher &publisher,const string &name, BALProblem &bal_problem);
#endif
#endif //CERES_LEARNING_UTILS_H

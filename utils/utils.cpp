//
// Created by libaoyu on 19-4-27.
//

#include "utils.h"
#include <Eigen/Core>
#include <Eigen/Dense>
void convert_cam_to_publis_format(double *state, double *published){
    Eigen::Map<Eigen::Vector3d> axisData(state);
    Eigen::AngleAxisd axisd(axisData.norm(),axisData/axisData.norm());
    Eigen::Quaterniond quaternion;
    quaternion = axisd;
    published[0] = quaternion.w();
    published[1] = quaternion.x();
    published[2] = quaternion.y();
    published[3] = quaternion.z();
    published[4] = state[3];
    published[5] = state[4];
    published[6] = state[5];
}
#ifdef ROS_DRAW
void draw_data(BAPublisher &publisher,const string &name, BALProblem &bal_problem){
    shared_ptr<ba_message_meta> message_meta_ptr = make_shared<ba_message_meta>(name,"red");
    publisher.addMessage(message_meta_ptr);
    double published_cam_pose[7];
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camState = bal_problem.mutable_point_for_pointidx(i);
        convert_cam_to_publis_format(camState,published_cam_pose);
        publisher.addQuaternionCameraToMessage(message_meta_ptr,published_cam_pose);
    }

    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = bal_problem.mutable_point_for_pointidx(i);
        publisher.addPointToMessage(message_meta_ptr,point);
    }
    publisher.show(message_meta_ptr);
}
#endif
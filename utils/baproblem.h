//
// Created by libaoyu on 19-4-24.
//

#ifndef CERES_LEARNING_BAPROBLEM_H
#define CERES_LEARNING_BAPROBLEM_H


// Read a Bundle Adjustment in the Large dataset.
#include <iostream>
#include <cassert>
#include <vector>
#include <set>
#ifdef ROS_DRAW
#include <BAPublisher.h>
#endif


class BALProblem {
public:
    ~BALProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }
    int num_observations()       const { return num_observations_;               }
    int num_cameras()       const { return num_cameras_;               }
    int num_points()       const { return num_points_;               }
    const double* observations() const { return observations_;                   }
    const double* observations(int i) const{return observations_ + i*2;}
    double* mutable_cameras()          { return parameters_;                     }
    double* mutable_points()           { return parameters_  + 9 * num_cameras_; }
    double* mutable_camera_for_cameraidx(int i) {
        return mutable_cameras() + i * 9;
    }
    double* mutable_point_for_pointidx(int i) {
        return mutable_points() + i * 3;
    }
    bool LoadFile(const char* filename) {
        FILE* fptr = fopen(filename, "r");
        if (fptr == nullptr) {
            return false;
        };
        FscanfOrDie(fptr, "%d", &num_cameras_);
        FscanfOrDie(fptr, "%d", &num_points_);
        FscanfOrDie(fptr, "%d", &num_observations_);
        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];
        num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
        parameters_ = new double[num_parameters_];
        for (int i = 0; i < num_observations_; ++i) {
            FscanfOrDie(fptr, "%d", camera_index_ + i);
            FscanfOrDie(fptr, "%d", point_index_ + i);
            for (int j = 0; j < 2; ++j) {
                FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
            }
        }
        for (int i = 0; i < num_parameters_; ++i) {
            FscanfOrDie(fptr, "%lf", parameters_ + i);
        }
        cam_res_set.resize(static_cast<unsigned long>(num_cameras()));
        point_res_set.resize(static_cast<unsigned long>(num_points()));
        for (int i = 0; i < num_observations(); ++i) {
            cam_res_set[camera_index_[i]].insert(i);
            point_res_set[point_index_[i]].insert(i);
        }
        cam_res_set_left = cam_res_set;
        point_res_set_left = point_res_set;
        return true;
    }
    void reset(){
        cam_res_set_left = cam_res_set;
        point_res_set_left = point_res_set;
    }
    int *getPoint_index_() const {
        return point_index_;
    }

    int *getCamera_index_() const {
        return camera_index_;
    }
    int getPoint_index_by_idx(int i) const {
        return point_index_[i];
    }

    int getCamera_index_by_idx(int i) const {
        return camera_index_[i];
    }
    const std::vector<std::set<int>> &get_cam_res_set_left() const {
        return cam_res_set_left;
    }

    const std::vector<std::set<int>> &get_point_res_set_left() const {
        return point_res_set_left;
    }
private:
    template<typename T>
    void FscanfOrDie(FILE *fptr, const char *format, T *value) {
        int num_scanned = fscanf(fptr, format, value);
//        if (num_scanned != 1) {
//            LOG(FATAL) << "Invalid UW data file.";
//        }
        assert(num_scanned == 1);
    }
    int num_cameras_{};
    int num_points_{};
    int num_observations_{};
    int num_parameters_{};
    int* point_index_=nullptr;
    int* camera_index_=nullptr;
    double* observations_=nullptr;
    double* parameters_=nullptr;
    std::vector<std::set<int>> cam_res_set;

    std::vector<std::set<int>> point_res_set;
    std::vector<std::set<int>> cam_res_set_left;
    std::vector<std::set<int>> point_res_set_left;
};

#endif //CERES_LEARNING_BAPROBLEM_H

//
// Created by libaoyu on 19-4-9.
//
#include <iostream>
#include <set>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <map>
#include "ResidualBA.h"
#include "HessionBA.h"

// BALProblem loader from ceres document
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
    double* mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * 9;
    }
    double* mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * 3;
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
    const vector<set<int>> &get_cam_res_set_left() const {
        return cam_res_set_left;
    }

    const vector<set<int>> &get_point_res_set_left() const {
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
    vector<set<int>> cam_res_set;

    vector<set<int>> point_res_set;
    vector<set<int>> cam_res_set_left;
    vector<set<int>> point_res_set_left;
};
class System{
public:
//    static Camera_t::Point_t::State
    static Camera_t::CamState double_to_state(Scalar *data){
        // camera[0,1,2] are the angle-axis rotation.
        // camera[3,4,5] are the translation.
        // 6 focal
        // 7,8 second and fourth order radial distortion.
        Eigen::Vector3d angleAxisData(data[0],data[1],data[2]);
        Eigen::Vector3d translation(data[3],data[4],data[5]);
        double radian = angleAxisData.norm();
        angleAxisData /= radian;
        Eigen::Quaterniond quaternion;
        quaternion = Eigen::AngleAxisd(radian,angleAxisData);

        Sophus::SE3<Scalar> se3(quaternion,translation);
        Camera_t::CamState ret;
        ret.block(0,0,6,1) = se3.log();
        ret(6) = data[6];
        ret(7) = data[7];
        ret(8) = data[8];
        return ret;
    }
    static Sophus::SE3<Scalar> double_to_se3(Scalar *data){
        Eigen::Vector3d angleAxisData(data[0],data[1],data[2]);
        Eigen::Vector3d translation(data[3],data[4],data[5]);
        double radian = angleAxisData.norm();
        angleAxisData /= radian;
        Eigen::Quaterniond quaternion;
        quaternion = Eigen::AngleAxisd(radian,angleAxisData);
        return Sophus::SE3<Scalar>(quaternion,translation);
    }
    bool get_point_in_window(BALProblem &balProblem, int window_start, int window_size, set<int> &result) {
        for (int i = 0; i < window_size; ++i) {//对窗口内每个相机
            int cam_idx = window_start + i;
            auto &cam_point_set = balProblem.get_cam_res_set_left()[cam_idx];
            for(auto iter = cam_point_set.begin(); iter != cam_point_set.end(); iter++){//对每个点
                //它的两个相机都在窗口内，则产生一个残差
                int res_idx = *iter;
                if(balProblem.getCamera_index_()[res_idx]){

                }
            }
        }
    }
    static bool check_recount(BALProblem &balProblem){
        map<int, int> datamap;
        for (int i = 0; i < balProblem.num_observations(); ++i) {
            int idx = balProblem.getCamera_index_()[i]
                      *balProblem.num_points()
                      + balProblem.getPoint_index_()[i];
            if(datamap.find(idx
                    )
                    != datamap.end()){//有重复，汇报情况！
                cout<<"recount finded!"
                << balProblem.getCamera_index_()[i]
                <<" "
                <<balProblem.getPoint_index_()[i]<<endl;

            }else{
                datamap.insert(make_pair(idx, i));
            }
        }
    }
    static bool check_point_res_seize(BALProblem &balProblem){
        for (int i = 0; i < balProblem.get_cam_res_set_left().size(); ++i) {
            auto &resset = balProblem.get_point_res_set_left()[i];
//            assert(resset.size()%2 == 0);
            if(resset.size() %2){
                cout<< resset.size() <<endl;
            }
        }
    }
};
void do_ba(WindowOptimizor_t &windowOptimizor, BALProblem &balProblem){
    int num_cams = balProblem.num_cameras();
    vector<Camera_t *> allCameras;
    for (int i = 0; i < num_cams; ++i) {
        Camera_t::CamState state = System::double_to_state(balProblem.mutable_camera_for_observation(i));
        auto *camera = new Camera_t(i,state);
        camera->se3State = System::double_to_se3(balProblem.mutable_camera_for_observation(i));
        allCameras.push_back(camera);
        windowOptimizor.insertCamera(camera);
    }
    vector<Point_t *> allPoints;
    for (int i = 0; i < balProblem.num_points(); ++i) {
//        if(i==0){
//            dbcout<<Eigen::Vector3d(balProblem.mutable_point_for_observation(i))<<endl;
//        }
        Point_t::PointState state;
        state<<balProblem.mutable_point_for_observation(i)[0],
        balProblem.mutable_point_for_observation(i)[1],
        balProblem.mutable_point_for_observation(i)[2];

        auto *point = new Point_t(i,state);
        allPoints.push_back(point);
        windowOptimizor.insertPoint(point);
    }
    cout<<"points all added"<<endl;

    vector<Residual_t *> allResiduals;
    for (int i = 0; i < balProblem.num_observations(); ++i) {
        auto *residual = new ResidualBA_t(
                allCameras[balProblem.getCamera_index_by_idx(i)],
                allPoints[balProblem.getPoint_index_by_idx(i)],
                ResidualBA_t::ResData(balProblem.observations(i)));
        allCameras[balProblem.getCamera_index_by_idx(i)]->getPoints().push_back(
                allPoints[balProblem.getPoint_index_by_idx(i)]);
        allResiduals.push_back(residual);
        windowOptimizor.addResidual(residual);
    }
    cout<<"residuals all added"<<endl;
    Mat delta;
    windowOptimizor.step_once(delta);
    cout<<"step once"<<endl;
}
int main(){
    BALProblem problemLoader;
    problemLoader.LoadFile("/home/libaoyu/Data/ba_problems/problem-49-7776-pre.txt");
    cout<<"problem loaded"<<endl;
//    cout<<problemLoader.num_observations()<<endl;
    int num_cams = problemLoader.num_cameras();
    HessionStructBA_t *hessionStructBA = new HessionStructBA_t();
    WindowOptimizor_t windowOptimizor(hessionStructBA);
    cout<<num_cams<<endl;
    cout<<problemLoader.num_observations()<<endl;
    do_ba(windowOptimizor,problemLoader);
//    auto data_list = problemLoader.get_cam_point_set();
//    for (int i = 0; i < data_list.size(); ++i) {
//        cout<<"camera "<<i<<" "<<data_list[i].size();
//        for (int j = 0; j < data_list[i].size(); ++j) {
////            cout<<data_list[i][j]<<", ";
//        }
//        cout<<endl;
//    }
}

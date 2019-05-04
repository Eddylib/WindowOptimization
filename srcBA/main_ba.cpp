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
#include "../utils/baproblem.h"
#include "../utils/utils.h"
// BALProblem loader from ceres document
class System{
public:
    static Scalar getTotalRes(vector<Residual_t *> &allres){
        Scalar ret = 0;
        for (int i = 0; i < allres.size(); ++i) {
            ret += allres[i]->asScalar();
        }
        return ret;
    }
};
void do_ba(WindowOptimizor_t &windowOptimizor, BALProblem &balProblem){
    int num_cams = balProblem.num_cameras();
#ifdef ROS_DRAW
    BAPublisher publisher = BAPublisher::createInstance("lby_ba",0,nullptr);
#endif
    vector<Camera_t *> allCameras;
    for (int i = 0; i < num_cams; ++i) {
        auto *camera = new Camera_t(i,
                balProblem.mutable_camera_for_cameraidx(i));
        allCameras.push_back(camera);
        windowOptimizor.insertCamera(camera);
    }
    vector<Point_t *> allPoints;
    for (int i = 0; i < balProblem.num_points(); ++i) {
        auto *point = new Point_t(i,balProblem.mutable_point_for_pointidx(i));
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
    }
    int cnt = 0;
    for (int i = 0; i < allPoints.size(); ++i) {
        cnt += allPoints[i]->getResiduals().size();
    }
    cout<<"residuals all added "<<allResiduals.size()<<" checked "<<cnt<<endl;
#ifdef ROS_DRAW
    draw_data(publisher,"lby/before",balProblem);
#endif
    Mat delta;
    double lr = 1.0;
    double min_res = std::numeric_limits<double>::max();
    for (int i = 0; i < 1000; ++i) {
        for (int j = 0; j < allResiduals.size(); ++j) {
            allResiduals[j]->computeRes();
        }
        double curr_res = System::getTotalRes(allResiduals);
        if(min_res > curr_res){
            min_res = curr_res;
#ifdef ROS_DRAW
            draw_data(publisher,"lby/after",balProblem);
#endif
        }
        cout<<"step curr residual "<<curr_res<<", min res "<<min_res<<endl;
        windowOptimizor.step_once(delta,true,true,true,true,log(i+1.0));
        //min res: 8.37132e+06-->3.96172e+05 残差减小20倍
    }
}
int main(){
    BALProblem problemLoader;
    problemLoader.LoadFile("/home/libaoyu/Data/ba_problems/problem-16-22106-pre.txt");
    cout<<"problem loaded"<<endl;
    int num_cams = problemLoader.num_cameras();
    HessionStructBA_t *hessionStructBA = new HessionStructBA_t();
    WindowOptimizor_t windowOptimizor(hessionStructBA);
    cout<<num_cams<<endl;
    cout<<problemLoader.num_observations()<<endl;
    do_ba(windowOptimizor,problemLoader);
}

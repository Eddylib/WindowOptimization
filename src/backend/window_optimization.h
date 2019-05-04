//
// Created by libaoyu on 18-10-31.
//

// 增加点，需要指定host，target，residual
//    void addPoint();
// 增加相机，需要指定一系列point
//    void addCamera();
// 增加residual，需要指定host，target，residual
//    void addResidual();

//marg点
//marg相机
//去除residual，（直接去除？）

// 初始化：如果没有相机，需要一个相机，初始为0,0,0，但是可用随机
// 对于点，在跟踪时点的深度会进行更新，收敛时成为成熟点进入到关键点集合，与一个host帧关联且与多个residual关联
// 每个residual与host和target关联，而这个关联应该就是在跟踪时建立的残差项，当有重投影误差时才能够关联，当成为成熟点时才加入优化系统
// 也就是说，点可以现有而不必要有residual，但必须有camera即host
// 当有residual时，必须指定host，target，以及point
//  增加顺序：camera-->point-->residual，
//  当camera有时，可以指定camera，也可以新建camera，
//  当camera满时还要新建camera就需要marg掉某个camera，或者直接指定camera增加点
//  点没有上限
//  当camera没有时，必须新建camera，同时新建一堆point，以后point可无序
//  需要定义当前的相机数目与点的数目进行marg更新相机位姿
// 当新的关键帧建立时，一些候选点被选择（即未成熟点），这些候选点在单目情况下是随机初始化的，在双目情况下有好的初始化
// 这些候选点被接下来的tracking共同refine（位姿加深度）
// 所有关键帧里既有成熟点也有未成熟点，跟踪时所有点都参与跟踪未成熟点与成熟点

// todo
// 接下来任务：
// done 1、优化代码实现窗口不满时的求解
// 2、实现增加相机，同时支持相机无点，有点无残差的系统计算（一般都是有点的吧？nonono，刚开始只有未成熟点）
        // 系统需要加入先验信息，先验信息即初始值，以及先验信息矩阵，凡点、相机加入系统都需要有这个信息！
        //
// 3、实现增加点，进一步支持有点无残差的计算（计算残差时忽略点即可）
// 4、实现增加残差，确保增加残差使得系统能够稳定运行
// 5、实现调度代码，动态增加点与残差

// 6、实现点的marg
// 7、实现camera的marg
// 8、实现调度代码，动态地增删点、相机。至此后端优化算法完成

// * 添加：优化结果保存在camera的结构内，而不是保存成一个数组，实现点的marg
// 未来计划，思考后端优化代码是否可彻底分离形成优化库，可否实现、开源

#pragma once

#include <cassert>
#include <vector>
#include <iostream>
#include <Eigen/Sparse>
using namespace std;
//#define DEBUG

#include "../alg_config.h"
#include "../alg_utils.h"
#include "../debug_utils.h"
#include "Point.h"
#include "Camera.h"
#include "Hession.h"
#include "Residual.h"

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
class WindowOptimizor{
public:
//    using Hxixi_t = Eigen::Matrix<SCALAR,WINDOW_SIZE_MAX*FRAME_DIM,WINDOW_SIZE_MAX*FRAME_DIM>;
    using HessionStruct_t = HessionBase<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>;
//    using Hxixi_t = Eigen::SparseMatrix<double>;
    using Camera_t =  Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>;
    using Point_t = Point<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> ;
    using Residual_t =  ResidualBase<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> ;

private:
    HessionStruct_t *hessionStruct;
//    Hxixi_t accB;// 位姿部分的hession矩阵维护在B中，此矩阵是jdrdx_{ij}^T*jdrdx_{ij}
    //上面两个矩阵的关系是一个仅对相对位姿求导（accB，对应于DSO中的EF什么什么中的acc）
    // 一个是绝对位姿（B对应于DSO中的EF什么什么中的Hm），相对位姿用于跟踪，绝对位姿用于建图
    void beginMargPoint();
    void endMarg();
    void marginlizeOnePoint(Point_t *point);
    //由于点被marg而导致的信息增值
    vector<Camera_t *> cameras;
    list<Point_t *> points;
    void addResidual(Residual_t *residual);
//    void attach_B();    //B 只有在第一次运算时才会调用此函数将所有数据加载到B上，之后只需进行优化来更新B
public:
    explicit WindowOptimizor(HessionStruct_t *_hessionStruct);
    const vector<Camera_t *> & getCameras()const{return cameras;}
    void insertCamera(Camera_t *camera);
    void insertPoint(Point_t *point);
    void step_once(
            Mat &delta,
            bool if_process_camera = true,
            bool if_process_points = false,
            bool if_update_camera = false,
            bool if_update_points = false,SCALAR lr=0.01);
    //marg掉一个点如何才算成功？ marg前后算位姿数据不变，这才说明信息被合理地添加到了相应的地方。
    // 外部代码只需要标记某个point为可marg，然后调用marginlizeFlaggedPoint即可marg掉点
    //之后需要调用clearMargedPoints来彻底删除marg掉的点。
    void marginlizeFlaggedPoint();
    void stepClear();
    void clearMargedPoints();
    void marginlizeOneCamera(Camera_t *camera);
    void resetHessionForStep(){hessionStruct->resetForStep();}
};

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::WindowOptimizor(
        HessionStruct_t *_hessionStruct) :hessionStruct(_hessionStruct){
            assert(hessionStruct && "hession struct donot initialized");
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::insertCamera(Camera_t *camera) {
    if(cameras.size() >= WINDOW_SIZE_MAX){
        cout<<__FUNCTION__<<", error reach max window size"<<endl;
        return;
    }
    cameras.push_back(camera);
}

//进行一次优化迭代，假定B已存在，任务： 将delta求解出（cameras，points里面都有）
//必然要更新B，更新JxR，ECinvw，
//如果要求点的更新量，EdertaXx要求出，点的derta会被更新
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void
WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::step_once(
        Mat &delta,
        bool if_process_camera,
        bool if_process_points,
        bool if_update_camera,
        bool if_update_points,SCALAR lr) {
    if(if_process_camera){
        dbcout<<"process camera"<<std::endl;
        stepClear();
        //只计算相机更新量
        int camSize = static_cast<int>(cameras.size());
        for(auto iter = points.begin(); iter!= points.end(); iter++){
//            dbcout<<(*iter)->getResiduals().size()<<endl;
            int sz = (*iter)->getResiduals().size();
            for (int i = 0; i < sz; ++i) {
                addResidual((*iter)->getResiduals()[i]);
            }
        }
        // hession的左上角B在addResidual中计算完成，并且右上角及右下角及等式右边的-Jr也完成
        // 这里开始根据舒尔补计算相机部分的-Jxr - EC^-1W
        for (int i = 0; i < camSize; ++i) {
            cameras[i]->einvCJpR = cameras[i]->einvCJpRMargedP;
        }
        for(auto iter = points.begin(); iter!= points.end(); iter++){
            Point_t *point = *iter;
            if(point->getStatus() == Status::MARGINALIZED)
                continue;
            assert(point->getStatus() != Status::TOBE_MARGINLIZE);
            hessionStruct->stepApplyPoint(point,cameras.size());
            // 准备方程组等号右边，上面的部分
            for (int k = 0; k < camSize; ++k) {
                if(point->hasResidualWithTarget(k))
                    cameras[k]->einvCJpR += point->getEik(k)
                                        *point->getC().inverse()*(-point->getJp_r());
            }
        }
        Mat optRight =
                Eigen::Matrix<SCALAR,WINDOW_SIZE_MAX*FRAME_DIM,1>::Ones();
        for (int i = 0; i < camSize; ++i) {
            optRight.block(i*FRAME_DIM,0,FRAME_DIM,1) = -cameras[i]->jx_r-cameras[i]->einvCJpR;;
//            dbcout<<cameras[i]->einvCJpR<<std::endl;
        }
        hessionStruct->solveStep(optRight,delta);
        if(if_update_camera){
            cout<<"update camera"<<std::endl;
            //改变相机状态
            for (int i = 0; i < camSize; ++i) {
                cameras[i]->applyDelta(lr,delta.block(i*FRAME_DIM,0,FRAME_DIM,1));
            }
        }
        //同时计算点的更新量
        if(if_process_points){
            cout<<"process point"<<std::endl;
            for(auto iter = points.begin(); iter!= points.end(); iter++){
                (*iter)->getDelta(delta);
            }
            if(if_update_points){
                cout<<"update point"<<std::endl;
                for(auto iter = points.begin(); iter!= points.end(); iter++){
                    (*iter)->applyDelta(lr);
                }
            }
        }
    }

}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::stepClear(){

    //清空相关变量以便下次迭代（只剩下点的状态，与导数相关的结构都需要清除）
    for (int i = 0; i < cameras.size(); ++i) {
        cameras[i]->clearStepInfo();
    }
    for(auto iter = points.begin(); iter!= points.end(); iter++){
        (*iter)->clearStepInfo();
    }
    hessionStruct->resetForStep();
}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::insertPoint(Point_t *point) {
    points.push_back(point);
}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::addResidual(Residual_t *residual) {
//    residual->computeRes();
    residual->applyDataToPoint(hessionStruct);
    hessionStruct->applyRes(residual);
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::beginMargPoint(){
    for (int i = 0; i < cameras.size(); ++i) {
        cameras[i]->einvCJpRMargedP.setZero();
    }
    hessionStruct->clearMargedInfo();
}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::marginlizeOnePoint(Point_t *point){
    //主要更新Camera的einvCJpRMargedP，
    // 和WindowOptimizor的BMargedP
    int camSize = static_cast<int>(cameras.size());
    //只要marg前后对相机位姿求解无影响即可认为marg成功
    for (int k = 0; k < camSize; ++k) {
        cameras[k]->einvCJpRMargedP -= point->getEik(k)
                                       *point->getC().inverse()*point->getJp_r();
    }
    hessionStruct->margOnePoint(point,cameras.size());
    point->setStatus(Status::MARGINALIZED);
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::marginlizeFlaggedPoint(){
    beginMargPoint();
    for (int i = 0; i < cameras.size(); ++i) {
        for (int j = 0; j < cameras[i]->getPoints().size(); ++j) {
            if(cameras[i]->getPoints()[j]->getStatus() == Status::TOBE_MARGINLIZE){
                marginlizeOnePoint(cameras[i]->getPoints()[j]);
            }
        }
    }
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::clearMargedPoints(){
    for (int i = 0; i < cameras.size(); ++i) {
        cout<<cameras[i]->getPoints().size()<<",";
    }
    cout<<endl;
    for (int i = 0; i < cameras.size(); ++i) {
        int point_size = cameras[i]->getPoints().size();
        for (int j = 0; j < point_size; ++j) {
            if(cameras[i]->getPoints()[j]->getStatus() == MARGINALIZED){
                Point_t* tmp = cameras[i]->getPoints()[j];
                cameras[i]->getPoints()[j] = cameras[i]->getPoints().back();
                cameras[i]->getPoints().pop_back();
                point_size--;
                delete tmp;
            }
        }
    }
    for (int i = 0; i < cameras.size(); ++i) {
        cout<<cameras[i]->getPoints().size()<<",";
    }
    cout<<endl;
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::marginlizeOneCamera(Camera_t *camera){

}

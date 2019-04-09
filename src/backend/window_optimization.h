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

using namespace std;
//#define DEBUG

#include "../alg_config.h"
#include "../alg_utils.h"
#include "../debug_utils.h"
#include "Point.h"
#include "Camera.h"
#include "Residual.h"
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
class WindowOptimizor{
public:
//    typedef Eigen::Matrix<SCALAR,WINDOW_SIZE_MAX*FRAME_DIM,WINDOW_SIZE_MAX*FRAME_DIM> Hxixi;
    using Hxixi = Eigen::Matrix<SCALAR,WINDOW_SIZE_MAX*FRAME_DIM,WINDOW_SIZE_MAX*FRAME_DIM>;
//    typedef Eigen::MatrixXd Hxixi;

    using Camera_t =  Camera<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>;
    using Point_t = Point<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> ;
    using Residual_t =  Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> ;
protected:
    Hxixi B;
    Hxixi accB;// 位姿部分的hession矩阵维护在B中，此矩阵是jdrdx_{ij}^T*jdrdx_{ij}
    //上面两个矩阵的关系是一个仅对相对位姿求导（accB，对应于DSO中的EF什么什么中的acc）
    // 一个是绝对位姿（B对应于DSO中的EF什么什么中的Hm），相对位姿用于跟踪，绝对位姿用于建图

    //由于点被marg而导致的信息增值
    Hxixi BMargedP;
    vector<Camera_t *> cameras;
    int bIsIdentity[WINDOW_SIZE_MAX];
    void beginMargPoint();
    void endMarg();
    void marginlizeOnePoint(Point_t *point);
public:
    WindowOptimizor();
    void attach_B();    //B 只有在第一次运算时才会调用此函数将所有数据加载到B上，之后只需进行优化来更新B
    vector<Camera_t *> & getCameras(){return cameras;}
    void insertCamera(Camera_t *camera);
    void insertPoint(Point_t *point);
    void step_once(
            Mat &delta,
            bool if_process_camera = true,
            bool if_process_points = false,
            bool if_update_camera = false,
            bool if_update_points = false);
    //marg掉一个点如何才算成功？ marg前后算位姿数据不变，这才说明信息被合理地添加到了相应的地方。
    void addResidual(Camera_t *_host, Camera_t *_target, Point_t *_point);
    // 外部代码只需要标记某个point为可marg，然后调用marginlizeFlaggedPoint即可marg掉点
    //之后需要调用clearMargedPoints来彻底删除marg掉的点。
    void marginlizeFlaggedPoint();
    void clearMargedPoints();
    void marginlizeOneCamera(Camera_t *camera);
};

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::attach_B() {
    B.setIdentity();
    accB.setIdentity();
    B.block(0,0,cameras.size()*FRAME_DIM,cameras.size()*FRAME_DIM).setZero();
    accB.block(0,0,cameras.size()*FRAME_DIM,cameras.size()*FRAME_DIM).setZero();
    for (int i = 0; i < cameras.size(); ++i) {
        auto &points = cameras[i]->getPoints();
        for (int j = 0; j < points.size(); ++j) {
            auto &reses = points[j]->getResiduals();
            for (int k = 0; k < reses.size(); ++k) {
                auto res = reses[k];
                int sth,stt;
                sth = FRAME_DIM*res->host->getid();
                stt = FRAME_DIM*res->target->getid();

                Mat accBTH = res->jdrdxi_th.transpose()*res->jdrdxi_th;

//                accB.block<FRAME_DIM,FRAME_DIM>(sth,sth) += accBTH;
                accB.block(sth,sth,FRAME_DIM,FRAME_DIM) += accBTH;
                B.block<FRAME_DIM,FRAME_DIM>(sth,sth) +=
                        res->getAdjH().transpose() *accBTH*res->getAdjH();

//                B.block<FRAME_DIM,FRAME_DIM>(stt,stt) += /*I*/accBTH/*I*/;

                B.block(stt,stt,FRAME_DIM,FRAME_DIM) += /*I*/accBTH/*I*/;
                B.block<FRAME_DIM,FRAME_DIM>(sth,stt) += res->getAdjH().transpose() *accBTH;/*I*/

                B.block<FRAME_DIM,FRAME_DIM>(stt,sth) += /*I*/accBTH *res->getAdjH();
            }
        }
    }
//    cout<<accB<<endl;
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::WindowOptimizor(){
    B.setIdentity();
    accB.setIdentity();
    for (int i = 0; i < WINDOW_SIZE_MAX; ++i) {
        bIsIdentity[i] = 1;
    }
    BMargedP.setZero();
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
        bool if_update_points) {
    if(if_process_camera){
        //只计算相机更新量
        Hxixi Bupdate = BMargedP;
        int camSize = static_cast<int>(cameras.size());
        int cntE = 0;
        for (int i = 0; i < camSize; ++i) {
//            cameras[i]->einvCJpR.setZero();
            cameras[i]->einvCJpR = cameras[i]->einvCJpRMargedP;
            //cameras[i]->jx_r与残差直接相关，归到addResidual里
            //cameras[i]->jx_r.setZero();
        }
        for (int i = 0; i < camSize; ++i) {
            Camera_t *camera = cameras[i];
            for (int j = 0; j < camera->getPoints().size(); ++j) {
                Point_t *point = camera->getPoints()[j];
                if(point->state == State::MARGINALIZED)
                    continue;
                //对每个点，统计信息到B
                //因为残差一变，就得重算Eik，根据总的Eik再重算Bundate
                //只能分成两步，先算Eik，在addresidual里
                //再算bupdate，在这里
                for (int k = 0; k < camSize; ++k) {
                    for (int l = 0; l < camSize; ++l) {
                        int str,stc;
                        str = k*FRAME_DIM;
                        stc = l*FRAME_DIM;
//                        Bupdate.block<FRAME_DIM,FRAME_DIM>(str,stc)+=
                        Bupdate.block(str,stc,FRAME_DIM,FRAME_DIM)+=
                                (*point->Eik[k])*point->C.inverse()*point->Eik[l]->transpose();
                    }
                }

                //对每个残差项
                //统计 sum{Eik * Ckk.inverse * Jx * r}，注意Jx为绝对坐标，包含了adjoint
                //归到addResidual里
//                for (int k = 0; k < point->getResiduals().size(); ++k) {
//                    Residual *res = point->getResiduals()[k];
//                    res->host->jx_r -= res->jthAdjH_r;
//                    res->target->jx_r -= res->jthAdjT_r;
//                    point->jp_r += res->jp_r;
//                }
                //jp_r应该乘以-1，所以这里变成了减等
                // 每个点对每个相机都有贡献！！
                for (int k = 0; k < camSize; ++k) {
                    cameras[k]->einvCJpR -= (*point->Eik[k])
                                            *point->C.inverse()*point->jp_r;
                }

//                cout<<__FUNCTION__<<","<<cntE++<<","<<(*point->Eik[i]).transpose()<<endl;
            }
        }
        Eigen::Matrix<SCALAR,WINDOW_SIZE_MAX*FRAME_DIM,1> optRight =
                Eigen::Matrix<SCALAR,WINDOW_SIZE_MAX*FRAME_DIM,1>::Ones();
#if 0
        int cnt = 0;
        Eigen::Matrix<SCALAR,WINDOW_SIZE_MAX*FRAME_DIM,1> VCam =
                Eigen::Matrix<SCALAR,WINDOW_SIZE_MAX*FRAME_DIM,1>::Ones();
        Eigen::Matrix<SCALAR,WINDOW_SIZE_MAX*FRAME_DIM,1> EcinvP =
                Eigen::Matrix<SCALAR,WINDOW_SIZE_MAX*FRAME_DIM,1>::Ones();
#endif
        //Bnew now is B, finished update
//        cout<<__FUNCTION__<<endl;
//        cout<<B<<endl;

        for (int i = 0; i < camSize; ++i) {
//            optRight.segment<FRAME_DIM>(i*FRAME_DIM) = cameras[i]->jx_r-cameras[i]->einvCJpR;
            optRight.block(i*FRAME_DIM,0,FRAME_DIM,1) = cameras[i]->jx_r-cameras[i]->einvCJpR;
#if 0
            VCam.segment<FRAME_DIM>(i*FRAME_DIM) = cameras[i]->jx_r;
            EcinvP.segment<FRAME_DIM>(i*FRAME_DIM) = cameras[i]->einvCJpR;
            cnt+=cameras[i]->getPoints().size();
#endif
        }
        delta = (B-Bupdate).colPivHouseholderQr().solve(optRight);
#if 0
        cout<<__FUNCTION__<<" Vcam"<<endl;
        cout<<VCam<<endl;
        cout<<__FUNCTION__<<" EcinvP"<<endl;
        cout<<EcinvP<<endl;
        Mat PointV;
        PointV.resize(cnt*POINT_DIM,1);
        int k = 0;
        for (int i = 0; i < camSize; ++i) {
            for (int j = 0; j < cameras[i]->getPoints().size(); ++j) {
                PointV.block(k*POINT_DIM,0,POINT_DIM,1) = cameras[i]->getPoints()[j]->jp_r;
                k++;
            }
        }
        cout<<__FUNCTION__<<" pointV"<<endl;
        cout<<PointV<<endl;
        cout<<__FUNCTION__<<" End"<<endl;
#endif
        if(if_update_camera){
            //改变相机状态
            B-=Bupdate;
        }
        //同时计算点的更新量
        if(if_process_points){
            if(if_update_points){

            }
        }
    }

}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::insertPoint(Point_t *point) {

}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::addResidual(Camera_t *_host, Camera_t *_target, Point_t *_point) {
    auto *residual = new Residual_t(_host,_target,_point);
    assert(_host && _target && _point);
    *_point->Eik[_host->getid()] += residual->getJthAdjH().transpose()*residual->jdrdp;
    *_point->Eik[_target->getid()] += residual->getJthAdjT().transpose()*residual->jdrdp;
    _point->getResiduals().push_back(residual);
    // one for target, one for host
    _point->C+= residual->jdrdp.transpose()*residual->jdrdp;
    residual->jthAdjH_r = residual->getJthAdjH().transpose()*residual->resdata;   //  jdrdxi_th * adjH *r 绝对位姿的导数乘以残差
    residual->jthAdjT_r = residual->getJthAdjT().transpose()*residual->resdata;   //  jdrdxi_th * adjT *r
    residual->jth_r = residual->jdrdxi_th.transpose()*residual->resdata;          //  jdrdxi_th *r
    residual->jp_r = residual->jdrdp.transpose()*residual->resdata;               //  jdrdp * r

    //B !!
//    cout<<_host->getid()<<","<<_target->getid()<<endl;

    int sth,stt;
    sth = FRAME_DIM*residual->host->getid();
    stt = FRAME_DIM*residual->target->getid();

    Mat accBTH = residual->jdrdxi_th.transpose()*residual->jdrdxi_th;

    if(bIsIdentity[residual->host->getid()]){
        accB.block(sth,sth,FRAME_DIM,FRAME_DIM).setZero();
        B.block(sth,sth,FRAME_DIM,FRAME_DIM).setZero();
//        accB.block<FRAME_DIM,FRAME_DIM>(sth,sth).setZero();
//        B.block<FRAME_DIM,FRAME_DIM>(sth,sth).setZero();
        bIsIdentity[residual->host->getid()] = 0;
    }
    if(bIsIdentity[residual->target->getid()]){
//        accB.block<FRAME_DIM,FRAME_DIM>(stt,stt).setZero();


//        B.block<FRAME_DIM,FRAME_DIM>(stt,stt).setZero();
        B.block(stt,stt,FRAME_DIM,FRAME_DIM).setZero();
        bIsIdentity[residual->target->getid()] = 0;
    }
//    accB.block<FRAME_DIM,FRAME_DIM>(sth,sth) += accBTH;
    accB.block(sth,sth,FRAME_DIM,FRAME_DIM) += accBTH;
//    B.block<FRAME_DIM,FRAME_DIM>(sth,sth) +=
    B.block(sth,sth,FRAME_DIM,FRAME_DIM) +=
            residual->getAdjH().transpose() *accBTH*residual->getAdjH();

//    B.block<FRAME_DIM,FRAME_DIM>(stt,stt) += /*I*/accBTH/*I*/;
    B.block(stt,stt,FRAME_DIM,FRAME_DIM) += /*I*/accBTH/*I*/;

//    B.block<FRAME_DIM,FRAME_DIM>(sth,stt) += residual->getAdjH().transpose() *accBTH;/*I*/
    B.block(sth,stt,FRAME_DIM,FRAME_DIM) += residual->getAdjH().transpose() *accBTH;/*I*/

//    B.block<FRAME_DIM,FRAME_DIM>(stt,sth) += /*I*/accBTH *residual->getAdjH();
    B.block(stt,sth,FRAME_DIM,FRAME_DIM) += /*I*/accBTH *residual->getAdjH();
    //other
    residual->host->jx_r -= residual->jthAdjH_r;
    residual->target->jx_r -= residual->jthAdjT_r;
    _point->jp_r += residual->jp_r;
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::beginMargPoint(){
    for (int i = 0; i < cameras.size(); ++i) {
        cameras[i]->einvCJpRMargedP.setZero();
    }
    BMargedP.setZero();
}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::marginlizeOnePoint(Point_t *point){
    //主要更新Camera的einvCJpRMargedP，
    // 和WindowOptimizor的BMargedP
    int camSize = static_cast<int>(cameras.size());
    //只要marg前后对相机位姿求解无影响即可认为marg成功
    for (int k = 0; k < camSize; ++k) {
        cameras[k]->einvCJpRMargedP -= (*point->Eik[k])
                                       *point->C.inverse()*point->jp_r;
    }
    for (int k = 0; k < camSize; ++k) {
        for (int l = 0; l < camSize; ++l) {
            int str,stc;
            str = k*FRAME_DIM;
            stc = l*FRAME_DIM;
//            BMargedP.block<FRAME_DIM,FRAME_DIM>(str,stc)+=
            BMargedP.block(str,stc,FRAME_DIM,FRAME_DIM)+=
                    (*point->Eik[k])*point->C.inverse()*point->Eik[l]->transpose();
        }
    }
    point->state = State::MARGINALIZED;
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::marginlizeFlaggedPoint(){
    beginMargPoint();
    for (int i = 0; i < cameras.size(); ++i) {
        for (int j = 0; j < cameras[i]->getPoints().size(); ++j) {
            if(cameras[i]->getPoints()[j]->state == State::TOBE_MARGINLIZE){
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
            if(cameras[i]->getPoints()[j]->state == MARGINALIZED){
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

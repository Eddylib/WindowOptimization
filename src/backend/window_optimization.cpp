//
// Created by libaoyu on 18-10-31.
//

#include "window_optimization.h"
#include <random>
#include <memory>
#include <iostream>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include "../../utils.h"

#include "Camera.h"
#include "Point.h"
#include "Residual.h"



void WindowOptimizor::attach_B() {
    B.setIdentity();
    accB.setIdentity();
    B.block(0,0,cameras.size()*FRAME_DIM,cameras.size()*FRAME_DIM).setZero();
    accB.block(0,0,cameras.size()*FRAME_DIM,cameras.size()*FRAME_DIM).setZero();
    for (int i = 0; i < cameras.size(); ++i) {
        auto &points = cameras[i]->getPoints();
        for (int j = 0; j < points.size(); ++j) {
            auto &reses = points[j]->getResiduals();
            for (int k = 0; k < reses.size(); ++k) {
                Residual *res = reses[k];
                int sth,stt;
                sth = FRAME_DIM*res->host->getid();
                stt = FRAME_DIM*res->target->getid();

                Mat accBTH = res->jdrdxi_th.transpose()*res->jdrdxi_th;

                accB.block<FRAME_DIM,FRAME_DIM>(sth,sth) += accBTH;

                B.block<FRAME_DIM,FRAME_DIM>(sth,sth) +=
                        res->getAdjH().transpose() *accBTH*res->getAdjH();

                B.block<FRAME_DIM,FRAME_DIM>(stt,stt) += /*I*/accBTH/*I*/;

                B.block<FRAME_DIM,FRAME_DIM>(sth,stt) += res->getAdjH().transpose() *accBTH;/*I*/

                B.block<FRAME_DIM,FRAME_DIM>(stt,sth) += /*I*/accBTH *res->getAdjH();
            }
        }
    }
//    cout<<accB<<endl;
}
WindowOptimizor::WindowOptimizor(){
    B.setIdentity();
    accB.setIdentity();
    for (int i = 0; i < WINDOW_SIZE_MAX; ++i) {
        bIsIdentity[i] = 1;
    }
    BMargedP.setZero();
}
void WindowOptimizor::insertCamera(Camera *camera) {
    if(cameras.size() >= WINDOW_SIZE_MAX){
        cout<<__FUNCTION__<<", error reach max window size"<<endl;
        return;
    }
    cameras.push_back(camera);
}

//进行一次优化迭代，假定B已存在，任务： 将delta求解出（cameras，points里面都有）
//必然要更新B，更新JxR，ECinvw，
//如果要求点的更新量，EdertaXx要求出，点的derta会被更新
void
WindowOptimizor::step_once(
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
            Camera *camera = cameras[i];
            for (int j = 0; j < camera->getPoints().size(); ++j) {
                Point *point = camera->getPoints()[j];
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
                        Bupdate.block<FRAME_DIM,FRAME_DIM>(str,stc)+=
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
            optRight.segment<FRAME_DIM>(i*FRAME_DIM) = cameras[i]->jx_r-cameras[i]->einvCJpR;
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

void WindowOptimizor::insertPoint(Point *point) {

}

void WindowOptimizor::addResidual(Camera *_host, Camera *_target, Point *_point) {
    Residual *residual = new Residual(_host,_target,_point);
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
        accB.block<FRAME_DIM,FRAME_DIM>(sth,sth).setZero();
        B.block<FRAME_DIM,FRAME_DIM>(sth,sth).setZero();
        bIsIdentity[residual->host->getid()] = 0;
    }
    if(bIsIdentity[residual->target->getid()]){
//        accB.block<FRAME_DIM,FRAME_DIM>(stt,stt).setZero();
        B.block<FRAME_DIM,FRAME_DIM>(stt,stt).setZero();
        bIsIdentity[residual->target->getid()] = 0;
    }
    accB.block<FRAME_DIM,FRAME_DIM>(sth,sth) += accBTH;
    B.block<FRAME_DIM,FRAME_DIM>(sth,sth) +=
            residual->getAdjH().transpose() *accBTH*residual->getAdjH();

    B.block<FRAME_DIM,FRAME_DIM>(stt,stt) += /*I*/accBTH/*I*/;

    B.block<FRAME_DIM,FRAME_DIM>(sth,stt) += residual->getAdjH().transpose() *accBTH;/*I*/

    B.block<FRAME_DIM,FRAME_DIM>(stt,sth) += /*I*/accBTH *residual->getAdjH();

    //other
    residual->host->jx_r -= residual->jthAdjH_r;
    residual->target->jx_r -= residual->jthAdjT_r;
    _point->jp_r += residual->jp_r;
}
void WindowOptimizor::beginMargPoint(){
    for (int i = 0; i < cameras.size(); ++i) {
        cameras[i]->einvCJpRMargedP.setZero();
    }
    BMargedP.setZero();
}

void WindowOptimizor::marginlizeOnePoint(Point *point){
    //主要更新Camera的einvCJpRMargedP，
    // 和WindowOptimizor的BMargedP
    int camSize = cameras.size();
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
            BMargedP.block<FRAME_DIM,FRAME_DIM>(str,stc)+=
                    (*point->Eik[k])*point->C.inverse()*point->Eik[l]->transpose();
        }
    }
    point->state = State::MARGINALIZED;
}
void WindowOptimizor::marginlizeFlaggedPoint(){
    beginMargPoint();
    for (int i = 0; i < cameras.size(); ++i) {
        for (int j = 0; j < cameras[i]->getPoints().size(); ++j) {
            if(cameras[i]->getPoints()[j]->state == State::TOBE_MARGINLIZE){
                marginlizeOnePoint(cameras[i]->getPoints()[j]);
            }
        }
    }
}
void WindowOptimizor::clearMargedPoints(){
    for (int i = 0; i < cameras.size(); ++i) {
        cout<<cameras[i]->getPoints().size()<<",";
    }
    cout<<endl;
    for (int i = 0; i < cameras.size(); ++i) {
        int point_size = cameras[i]->getPoints().size();
        for (int j = 0; j < point_size; ++j) {
            if(cameras[i]->getPoints()[j]->state == MARGINALIZED){
                Point* tmp = cameras[i]->getPoints()[j];
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
void WindowOptimizor::marginlizeOneCamera(Camera *camera){

}


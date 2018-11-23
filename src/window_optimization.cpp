//
// Created by libaoyu on 18-10-31.
//

#include "window_optimization.h"
#include <random>
#include <memory>
#include <iostream>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include "../utils.h"

#include "Camera.h"
#include "Point.h"
#include "Residual.h"
void OptimizerDebugC::reset() {
    init(0,0);
}
int get_rand(size_t max,int nodiff = -1){
    int ret = static_cast<int>(rand() % max);
    if(nodiff >= 0){
        while(ret == nodiff){
            ret = static_cast<int>(rand() % max);
        }
    }
    return ret;
}
void OptimizerDebugC::init(int camNum, int resNum) {
    srand(100);
    for (int i = 0; i < camNum; ++i) {
        cameras.push_back(new Camera(i));
    }
    vector<int> camSeleted(cameras.size(),0);
    for (int i = 0; i < resNum; ++i) {
        // 随机选择两个帧，新建残差
        int hostCamSelected = get_rand(cameras.size());
        int targetCamSelected = get_rand(cameras.size(),hostCamSelected);

        Camera *host = cameras[hostCamSelected];
        Camera *target = cameras[targetCamSelected];
        newResidual(host,target);
        camSeleted[hostCamSelected] = 1;
    }
    for (int i = 0; i < camSeleted.size(); ++i) {
        if(!camSeleted[i]){
            // 某个帧完全没有残差，强制生成
            int hostCamSelected = i;
            int targetCamSelected = get_rand(cameras.size(),hostCamSelected);

            Camera *host = cameras[hostCamSelected];
            Camera *target = cameras[targetCamSelected];
            newResidual(host,target);
            camSeleted[i] = 1;
        }
    }
    // 整合相机信息到B中
    attach_B();
}

Residual *OptimizerDebugC::newResidual(Camera *host, Camera *target, float possibilityCreateNewPoint) {
    Point *point = nullptr;
    int ifcreate = float(rand()%100)/100.f > (1.f-possibilityCreateNewPoint);
    float tmp = float(rand()%100)/100.f;
//    cout<<ifcreate<<endl;
    // host没点，必然要新建点，possibilityCreateNewPoint新建点
    if(host->getPoints().empty() || ifcreate){
        point = host->newPoint();
    }
    // 如果没有新建点，则在host里选择一个点，如果选的点与target和host已经有联系，则再选则其他的与host无联系的点
    // 如果找不到，才新建点
    if(!point){
        Point *tmp = nullptr;
        if(!target->getPoints().empty()){
            int ptIdx = get_rand(target->getPoints().size());
            tmp = target->getPoints()[ptIdx];
        }
        if(!tmp || tmp->hasResidualWithTarget(host->getid())){
            tmp = host->newPoint();
        }
        point = tmp;
    }
    Residual *residual = new Residual(host,target,point);
    return residual;
}

void OptimizerDebugC::printInfo() {
    int cameraCnt = 0;
    int pointCnt = 0;
    int resCnt = 0;
    fetchInfo(&cameraCnt,&pointCnt,&resCnt);
    cout<<"camera num: "<<cameraCnt<<"\npoint num: "<<pointCnt<<"\nresidual num: "<<resCnt<<"*2"<<endl;
//    for (int i = 0; i < cameras.size(); ++i) {
//        cout<<"******************************"<<endl;
//        Camera *camera = cameras[i];
//        cout<<"camera: "<<camera->getid()<<" has "<<camera->getPoints().size()<<" points"<<endl;
//        for (int j = 0; j < camera->getPoints().size(); ++j) {
//            Point *point = camera->getPoints()[j];
//            cout<<"--------------------------------------"<<endl;
//            cout<<"point:"<<point->getId()<<" has "<<point->getResiduals().size()<<"res"<<endl;
//            for (int k = 0; k < WINDOW_SIZE_MAX; ++k) {
//                cout<<"Eik:....................."<<endl;
//                cout<<point->Eik[k]<<endl;
//            }
//            cout<<"C:....................."<<endl;
//            cout<<point->C<<endl;
//            for (int k = 0; k < point->getResiduals().size(); ++k) {
//                Residual *residual = point->getResiduals()[k];
//                cout<<"residual: "<< residual->host->getid()<<","<<residual->target->getid()<<endl;
//            }
//        }
//    }
    cout<<"******************"<<endl;
}

void OptimizerDebugC::fetchInfo(int *camNum, int *pointNum, int *resNum) {
    int cameraCnt = 0;
    int pointCnt = 0;
    int resCnt = 0;
    for (int i = 0; i < cameras.size(); ++i) {
        auto &pts = cameras[i]->getPoints();
        cameraCnt ++;
        pointCnt += pts.size();
        for (int j = 0; j < pts.size(); ++j) {
            auto &res = pts[j]->getResiduals();
            resCnt += res.size();
        }
    }
    if(camNum){
        *camNum = cameraCnt;
    }
    if(pointNum){
        *pointNum = pointCnt;
    }
    if(resNum){
        *resNum = resCnt;
    }
}

Mat OptimizerDebugC::generateVCam() {
    // V: -J*r
    int resCnt = 0;
    vector<Point*> allPoints = getAllPoints();
    Mat ret;
    int matSize =FRAME_DIM * cameras.size();
    ret.resize(matSize,1);
    ret.setZero();
    for (int i = 0; i < allPoints.size(); ++i) {
        Point *point = allPoints[i];
        for (int j = 0; j < point->getResiduals().size(); ++j) {
            Residual *res = point->getResiduals()[j];
            int str,stc;
            stc = 0;
            str = FRAME_DIM*res->host->getid();
            ret.block(str,stc,FRAME_DIM,1) -= res->getJthAdjH().transpose()*res->resdata;
            str = FRAME_DIM*res->target->getid();
            ret.block(str,stc,FRAME_DIM,1) -= res->getJthAdjT().transpose()*res->resdata;
        }
    }
//    cout<<__FUNCTION__<<endl;
//    cout<<ret<<endl;
    return ret;
}

Mat OptimizerDebugC::generateVPoint() {
    // V: -J*r
    vector<Point*> allPoints = getAllPoints();
    Mat ret;
    int matSize = static_cast<int>(POINT_DIM * allPoints.size());
    ret.resize(matSize,1);
    ret.setZero();
    for (int i = 0; i < allPoints.size(); ++i) {
        Point *point = allPoints[i];
        for (int j = 0; j < point->getResiduals().size(); ++j) {
            Residual *res = point->getResiduals()[j];
            int str,stc;
            stc = 0;
            str = POINT_DIM*i;
            ret.block(str,stc,POINT_DIM,1) -= res->jdrdp.transpose()*res->resdata;
        }
    }
    return ret;
}

Mat OptimizerDebugC::generateTotalV() {
    // V: -J*r
    int resCnt = 0;
    vector<Point*> allPoints = getAllPoints();
    Mat ret;
    int matSize = static_cast<int>(FRAME_DIM * cameras.size() + POINT_DIM * allPoints.size());
    ret.resize(matSize,RES_DIM);
    ret.setZero();
    int str,stc;
    // camera part
    str = 0; stc = 0;
    ret.block(str,stc,cameras.size()*FRAME_DIM,1) += generateVCam();
    // point part
    str = cameras.size()*FRAME_DIM; stc = 0;
    ret.block(str,stc,POINT_DIM * allPoints.size(),1) += generateVPoint();

    return ret;
}
Mat OptimizerDebugC::generateTotalH() {
    auto J = generateTotalJ();
    auto ret =J.transpose()*J;
    vector<Point *> allPoints = getAllPoints();
#ifdef DEBUG
    cout<<__FUNCTION__<<", E"<<endl;
    cout<<ret.block(0,cameras.size()*FRAME_DIM,cameras.size()*FRAME_DIM,allPoints.size()*POINT_DIM)<<endl;
    auto HE = ret.block(0,cameras.size()*FRAME_DIM,cameras.size()*FRAME_DIM,allPoints.size()*POINT_DIM);
    auto HB = ret.block(0,0,cameras.size()*FRAME_DIM,cameras.size()*FRAME_DIM);
    auto HC = ret.block(cameras.size()*FRAME_DIM,cameras.size()*FRAME_DIM,allPoints.size()*POINT_DIM,allPoints.size()*POINT_DIM);
    cout<<__FUNCTION__<<", update"<<endl;
    cout<<HE*HC*HE.transpose()<<endl;
    cout<<__FUNCTION__<<", C"<<endl;
    cout<<HC<<endl;
    cout<<__FUNCTION__<<", E Cinv W"<<endl;
    cout<<HE*HC.inverse()*generateVPoint()<<endl;
#endif
    return ret;
}

vector<Point *> OptimizerDebugC::getAllPoints() {
    vector<Point *> ret;
    int cameraCnt = 0;
    int pointCnt = 0;
    int resCnt = 0;
    fetchInfo(&cameraCnt,&pointCnt,&resCnt);
    for (int i = 0; i < cameraCnt; ++i) {
        Camera *camera = cameras[i];
        for (int j = 0; j < camera->getPoints().size(); ++j) {
            Point *point = camera->getPoints()[j];
            ret.push_back(point);
        }
    }
    return ret;
}

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
        Hxixi Bupdate = Hxixi::Zero();
        int camSize = static_cast<int>(cameras.size());
        for (int i = 0; i < camSize; ++i) {
            cameras[i]->einvCJpR.setZero();
            cameras[i]->jx_r.setZero();
        }
        int cntE = 0;
        for (int i = 0; i < camSize; ++i) {
            Camera *camera = cameras[i];
            for (int j = 0; j < camera->getPoints().size(); ++j) {
                Point *point = camera->getPoints()[j];
                point->jp_r.setZero();
            }
            for (int j = 0; j < camera->getPoints().size(); ++j) {
                Point *point = camera->getPoints()[j];
                //对每个点，统计信息到B
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
                for (int k = 0; k < point->getResiduals().size(); ++k) {
                    Residual *res = point->getResiduals()[k];
                    res->host->jx_r -= res->jthAdjH_r;
                    res->target->jx_r -= res->jthAdjT_r;
                    point->jp_r += res->jp_r;
                }
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

Mat OptimizerDebugC::genBUpdate() {
    Hxixi ret;
    ret.setZero();
    // Bnew = B - E * C^-1 * Et
    vector<Point *> allPoints = getAllPoints();
    for (int i = 0; i < cameras.size(); ++i) {
        for (int j = 0; j < cameras.size(); ++j) {
            int str,stc;
            str = i*FRAME_DIM;
            stc = j*FRAME_DIM;
            for (int k = 0; k < allPoints.size(); ++k) {
                Point *point = allPoints[k];
                ret.block(str,stc,FRAME_DIM,FRAME_DIM) +=
                        (*point->Eik[i])*point->C.inverse()*point->Eik[j]->transpose();
            }
        }
    }

#ifdef DEBUG
    cout<<__FUNCTION__<<", E"<<endl;
    Mat E;
    E.resize(cameras.size()*FRAME_DIM,allPoints.size()*POINT_DIM);
    E.setZero();
    for (int i = 0; i < cameras.size(); ++i) {
        for (int j = 0; j < allPoints.size(); ++j) {
            E.block(i*FRAME_DIM,j*POINT_DIM,FRAME_DIM,POINT_DIM) += allPoints[j]->Eik[i];
        }
    }
    cout<<E<<endl;
    cout<<__FUNCTION__<<", update"<<endl;
    cout<<ret.block(0,0,cameras.size()*FRAME_DIM,cameras.size()*FRAME_DIM)<<endl;
    cout<<__FUNCTION__<<", C"<<endl;
    Mat C;
    C.resize(allPoints.size()*POINT_DIM,allPoints.size()*POINT_DIM);
    C.setZero();
    for (int i = 0; i < allPoints.size(); ++i) {
        C.block(i*POINT_DIM,i*POINT_DIM,POINT_DIM,POINT_DIM) = allPoints[i]->C;
    }
    cout<<C<<endl;
#endif
    return ret.block(0,0,cameras.size()*FRAME_DIM,cameras.size()*FRAME_DIM);
}

Mat OptimizerDebugC::genE_CInv_VPoint(const Mat &vPoint) {
    Mat ret;
    vector<Point *> allPoints = getAllPoints();
    ret.resize(cameras.size()*FRAME_DIM,1);
    ret.setZero();
//    int cnt = 0;
    for (int i = 0; i < cameras.size(); ++i) {
        for (int j = 0; j < allPoints.size(); ++j) {
            Point*point = allPoints[j];
            int str,stc;
            str = i*FRAME_DIM;
            stc = 0;
            ret.block(str,stc,FRAME_DIM,1) += (*point->Eik[i])*point->C.inverse()*vPoint.block(j*POINT_DIM,0,POINT_DIM,1);
//            cout<<__FUNCTION__<<","<<cnt++<<","<<(*point->Eik[i]).transpose()<<endl;
        }

    }

#ifdef DEBUG
    cout<<__FUNCTION__<<"E Cinv V"<<endl;
    cout<<ret<<endl;
#endif
    return ret;
}

Mat OptimizerDebugC::generateTotalJ() {
    Mat ret;
    vector<Point*> points = getAllPoints();
    vector<Residual*> residuals = getAllResiduals();
    ret.resize(residuals.size()*RES_DIM,cameras.size()*FRAME_DIM+POINT_DIM*points.size());
    ret.setZero();
    for (int i = 0; i < residuals.size(); ++i) {
        Residual *res = residuals[i];
        int str,stc;
        str = i*RES_DIM;
        stc = res->host->getid()*FRAME_DIM;
        ret.block(str,stc,RES_DIM,FRAME_DIM) += res->getJthAdjH();

        stc = res->target->getid()*FRAME_DIM;
        ret.block(str,stc,RES_DIM,FRAME_DIM) += res->getJthAdjT();

        int ptid = -1;
        for (int j = 0; j < points.size(); ++j) {
            if(res->point == points[j]){
                ptid = j;
                break;
            }
        }
        assert(ptid>=0);
        stc = cameras.size()*FRAME_DIM + ptid*POINT_DIM;

        ret.block(str,stc,RES_DIM,POINT_DIM) += res->jdrdp;
    }
    return ret;
}

vector<Residual *> OptimizerDebugC::getAllResiduals() {
    vector<Residual *> ret;
    vector<Point*> points = getAllPoints();
    for (int i = 0; i < points.size(); ++i) {
        ret.insert(ret.end(),points[i]->getResiduals().begin(),points[i]->getResiduals().end());
    }
    return ret;
}

Mat OptimizerDebugC::generateTotalR() {
    Mat ret;
    vector<Residual *> residuals = getAllResiduals();
    ret.resize(residuals.size()*RES_DIM,1);
    ret.setZero();

    for (int i = 0; i < residuals.size(); ++i) {
        Residual* res = residuals[i];
        int str,stc;
        stc = 0;
        str = i*RES_DIM;
        ret.block(str,stc,RES_DIM,1) = res->resdata;
    }
    return ret;
}

Mat OptimizerDebugC::generateCamUpdate() {
//    Mat update = genBUpdate();
    Mat bNew = get_B() - genBUpdate();
    Mat eCinvVPoint = genE_CInv_VPoint(generateVPoint());
#if 0
    cout<<__FUNCTION__<<"Vcam"<<endl;
    cout<<generateVCam()<<endl;
    cout<<__FUNCTION__<<"eCinvVPoint"<<endl;
    cout<<eCinvVPoint<<endl;
    cout<<__FUNCTION__<<"generateVPoint"<<endl;
    cout<<generateVPoint()<<endl;
    cout<<__FUNCTION__<<"end"<<endl;
#endif
    Mat xMargSolveCam = bNew.bdcSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(generateVCam()-eCinvVPoint);
    return xMargSolveCam;
}



void test_init(OptimizerDebugC &system){

//    system.printInfo();
#ifdef DEBUG
    cout<<__FUNCTION__<<", ttj"<<endl;
    cout<<ttj<<endl;
#endif

//    cout<<B-tth.block(0,0,system.getCameras().size()*FRAME_DIM,system.getCameras().size()*FRAME_DIM)<<endl;
//
//    cout<<"*********************"<<__FUNCTION__<<" result"<<endl;
//    cout<< system.generateCamUpdate()<<endl;
#ifdef DEBUG
    Mat ttj = system.generateTotalJ();
    Mat ttr = system.generateTotalR();
    Mat B = system.get_B();

#endif
    timeval t[2];
    start(t);
    auto margsolve =system.generateCamUpdate();
    stop(t);
    cout<<"marg solve calculate duration: "<<duration(t)<<"ms"<<endl;

    Mat delta;
    start(t);
    system.step_once(delta);
    stop(t);
    cout<<"split marg calculate duration: "<<duration(t)<<"ms"<<endl;


    cout<<"testing now>>"<<__FUNCTION__<<endl;
    Mat tth = system.generateTotalH();
    Mat ttv = system.generateTotalV();
    start(t);
    Mat xDirectSolve = tth.colPivHouseholderQr().solve(ttv);
    stop(t);
    Mat xDirectSolveCam = xDirectSolve.block(0,0,FRAME_DIM*system.getCameras().size(),1);
    cout<<"direct solve calculate duration: "<<duration(t)<<"ms"<<endl;

    cout<<"*********************"<<__FUNCTION__<<" errors in direct and marg sove: rows, cols, rmse: "<<endl;
    Mat error = xDirectSolveCam - margsolve;
    Mat rmse = error.transpose()*error;
    cout<<error.rows()<<", "<<error.cols()<<", "<<rmse(0)/(xDirectSolveCam.transpose()*xDirectSolveCam)(0)<<endl;

    cout<<"*********************"<<__FUNCTION__<<" errors in direct and split marg sove: rows, cols, rmse: "<<endl;
    Mat error2 = xDirectSolveCam - delta.block(0,0,xDirectSolveCam.rows(),1);
    Mat rmse2 = error2.transpose()*error2;
    cout<<delta.rows()<<", "<<delta.cols()<<", "<<rmse2(0)/(xDirectSolveCam.transpose()*xDirectSolveCam)(0)<<endl;
    cout<<"************************************************************************************"<<endl;
}

void test_add_camera(OptimizerDebugC &system){
    system.insertCamera(new Camera(static_cast<int>(system.getCameras().size())));
}
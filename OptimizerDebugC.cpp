//
// Created by libaoyu on 19-1-21.
//

#include "OptimizerDebugC.h"
#include "src/backend/Camera.h"
#include "src/backend/Point.h"
#include "src/backend/Residual.h"
#include "src/debug_utils.h"

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
    srand(11);
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
#if 0
    cout<<__FUNCTION__<<endl;
    auto oldB = B;
    cout<<"-------------------"<<endl;
    cout<<oldB<<endl;
    attach_B();
    cout<<"-------------------"<<endl;
    cout<<B<<endl;
    cout<<"-------------------"<<endl;
    cout<<B-oldB<<endl;
#endif
}

void OptimizerDebugC::newResidual(Camera *host, Camera *target, float possibilityCreateNewPoint) {
    Point *point = nullptr;
    int ifcreate = float(rand()%100)/100.f > (1.f-possibilityCreateNewPoint);
//    cout<<ifcreate<<endl;
    // host没点，必然要新建点，possibilityCreateNewPoint新建点
    if(host->getPoints().empty() || ifcreate){
        point = host->newPoint();
    }
    // 如果没有新建点，则在host里选择一个点，如果选的点与target和host已经有联系，则再选则其他的与host无联系的点
    // 如果找不到，才新建点
    if(!point){
        Point *tmp = nullptr;
        int ptIdx = get_rand(host->getPoints().size());
        tmp = host->getPoints()[ptIdx];
        if(!tmp || tmp->hasResidualWithTarget(target->getid())){
            tmp = host->newPoint();
        }
        point = tmp;
    }
    addResidual(host,target,point);
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
//    vector<Point *> allPoints = getAllPoints();
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

void OptimizerDebugC::reset() {
    init(0,0);
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


    cout<<"************************************************************************************"<<endl;
    timeval t[2];
    cout<<"testing now>>"<<__FUNCTION__<<endl;
    Mat tth = system.generateTotalH();
    Mat ttv = system.generateTotalV();
    start(t);
    Mat xDirectSolve = tth.colPivHouseholderQr().solve(ttv);
    stop(t);
    Mat xDirectSolveCam = xDirectSolve.block(0,0,FRAME_DIM*system.getCameras().size(),1);
    cout<<"direct solve calculate duration: "<<duration(t)<<"ms"<<endl;

    start(t);
    auto margsolve =system.generateCamUpdate();
    stop(t);
    cout<<"marg solve calculate duration: "<<duration(t)<<"ms"<<endl;

    Mat delta;
    start(t);
    system.clearMargedPoints();
    system.step_once(delta);
    stop(t);
    cout<<"split marg calculate duration: "<<duration(t)<<"ms"<<endl;

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

void test_marg_point(OptimizerDebugC &system, float possibility) {
    assert(possibility < 1);
    vector<Camera *> &cameras = system.getCameras();
    int camSize = cameras.size();
    for (int i = 0; i < camSize; ++i) {
        if(cameras[i]->getPoints().size() <= 1) continue;
        for (int j = 0; j < cameras[i]->getPoints().size(); ++j) {
            int ifmarg = float(rand()%100)/100.f>(1.f-possibility);
            if(ifmarg){
                cameras[i]->getPoints()[j]->state = State::TOBE_MARGINLIZE;
            }
        }
    }
    system.marginlizeFlaggedPoint();
}
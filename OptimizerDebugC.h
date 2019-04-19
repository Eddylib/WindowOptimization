//
// Created by libaoyu on 19-1-21.
//

#ifndef WINDOWOPTIMIZATION_OPTIMIZERDEBUGC_H
#define WINDOWOPTIMIZATION_OPTIMIZERDEBUGC_H

#include "src/backend/window_optimization.h"
//用于BA的残差结构与用于前端优化的残差结构不同
//前端滑窗优化的res有着target和host的概念，位姿求导分别要求左右两个相机的倒数
//而BA中一个res只与一个相机关联，所以应该是无法套用前端的滑窗优化的
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
class Residual:public ResidualBase<RES_DIM, FRAME_DIM, WINDOW_SIZE_MAX, POINT_DIM, SCALAR>{
public:
    // apply data while init
    using Base_t = ResidualBase<RES_DIM, FRAME_DIM, WINDOW_SIZE_MAX, POINT_DIM, SCALAR>;
    using WindowOptimizor_t = typename Base_t::WindowOptimizor_t;
    using Camera_t = typename Base_t::Camera_t;
    using Point_t = typename Base_t::Point_t;
    using Jdrdp_t = typename Base_t ::Jdrdp_t;
    using Jdrdxi_t = typename Base_t ::Jdrdxi_t;
    using Adjoint = typename Base_t ::Adjoint;
    using HessionBase_t = HessionBase<RES_DIM, FRAME_DIM, WINDOW_SIZE_MAX, POINT_DIM, SCALAR>;

    void initApplyData(HessionBase_t *hessionBase);
    Residual(Camera_t *_host,Camera_t *_target,Point_t *_point,Eigen::Matrix<SCALAR,RES_DIM,1>);
    Residual(Camera_t *_host,Camera_t *_target,Point_t *_point);
    Adjoint getAdjH();
    Adjoint getAdjT();
    Jdrdxi_t getJthAdjH();
    Jdrdxi_t getJthAdjT();
    Camera_t *getHost(){return host;}
    Camera_t *getTarget(){return target;}
    Point_t *getPoint(){return point;}
    const Eigen::Matrix<SCALAR,RES_DIM,1> &getResdata() const {return resdata;}
    const Jdrdp_t &getJdrdp(){return jdrdp;}

    void computeRes() override {};
    void computeJ() override {};
    Jdrdxi_t jdrdxi_th; //相对位姿
private:
    Camera_t *host;
    Camera_t *target;
    Point_t *point;
    Jdrdp_t jdrdp;
    Eigen::Matrix<SCALAR,RES_DIM,1> resdata;
    Eigen::Matrix<SCALAR,FRAME_DIM,1> jthAdjH_r;   //  jdrdxi_th' * adjH *r 绝对位姿的导数乘以残差
    Eigen::Matrix<SCALAR,FRAME_DIM,1> jthAdjT_r;   //  jdrdxi_th' * adjT *r
    Eigen::Matrix<SCALAR,FRAME_DIM,1> jth_r;       //  jdrdxi_th' *r
    Eigen::Matrix<SCALAR,POINT_DIM,1> jp_r;        //  jdrdp' *r
};



template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Residual(Camera_t *_host, Camera_t *_target, Point_t *_point):
        Residual(_host,_target,_point,DataGenerator::gen_data<Eigen::Matrix<SCALAR,RES_DIM,1>>()){
    //DataGenerator::gen_data<Eigen::Matrix<SCALAR,RES_DIM,1>>()
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Residual(Camera_t *_host,Camera_t *_target,Point_t *_point,Eigen::Matrix<SCALAR,RES_DIM,1> _resdata):
        host(_host),target(_target),point(_point),
        jdrdp(DataGenerator::gen_data<Jdrdp_t>()),
        jdrdxi_th(DataGenerator::gen_data<Jdrdxi_t>()),
        resdata(_resdata){
    if(!(host && target && point)){
//        cerr<<"warning: residual"
    }

    jthAdjH_r = getJthAdjH().transpose()*resdata;   //  jdrdxi_th * adjH *r 绝对位姿的导数乘以残差
    jthAdjT_r = getJthAdjT().transpose()*resdata;   //  jdrdxi_th * adjT *r
    jth_r = jdrdxi_th.transpose()*resdata;          //  jdrdxi_th *r
    jp_r = jdrdp.transpose()*resdata;


    //other
    if(host)
        host->jx_r -= jthAdjH_r;
    if(target)
        target->jx_r -= jthAdjT_r;
    if(_point)
        _point->addJp_r(jp_r);
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
typename Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Adjoint
Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::getAdjH() {
    return host->getAdjointAsH();
}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
typename Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Adjoint
Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::getAdjT() {
    return target->getAdjointAsT();
}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
typename Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Jdrdxi_t
Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::getJthAdjH() {
    return jdrdxi_th*getAdjH();
}
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
typename  Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::Jdrdxi_t
Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>::getJthAdjT() {
    return jdrdxi_th;/* *I */
}

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
void
Residual<RES_DIM, FRAME_DIM, WINDOW_SIZE_MAX, POINT_DIM, SCALAR>::initApplyData(
        HessionBase_t *hessionBase) {
    // one for target, one for host
    point->addEik(host->getid(),getJthAdjH().transpose()*jdrdp);
    point->addEik(target->getid(),getJthAdjT().transpose()*jdrdp);
    point->getResiduals().push_back(this);
    point->addC(jdrdp.transpose()*jdrdp);             //  jdrdp * r
    hessionBase->initapplyRes(this);
}


template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
class Hession:public HessionBase<RES_DIM, FRAME_DIM, WINDOW_SIZE_MAX, POINT_DIM, SCALAR> {
public:
    using Hxixi_t = Eigen::Matrix<SCALAR, WINDOW_SIZE_MAX * FRAME_DIM, WINDOW_SIZE_MAX * FRAME_DIM>;
    using HessionBase_t = HessionBase<RES_DIM, FRAME_DIM, WINDOW_SIZE_MAX, POINT_DIM, SCALAR>;
    using Residual_t = Residual<RES_DIM, FRAME_DIM, WINDOW_SIZE_MAX, POINT_DIM, SCALAR>;
    using ResidualBase_t = typename HessionBase_t::ResidualBase_t;
    using Point_t = typename HessionBase_t::Point_t;
    using Camera_t = typename HessionBase_t::Camera_t;
    Hxixi_t b;
private:
    Hxixi_t BMargedP;
    Mat Bupdate;
    //当该框无相机时，H的相应位置应该被设置为单位矩阵
    int bIsIdentity[WINDOW_SIZE_MAX] = {0};

    bool ifCamUninit(int idx) { return bIsIdentity[idx]; }

    void setCaminited(int idx) { bIsIdentity[idx] = 0; }


    void setZeroBpart(int str, int stc, int cols, int rows) {
        b.block(str, stc, cols, rows).setZero();
    }

public:

    Hession() {
        b.resize(WINDOW_SIZE_MAX * FRAME_DIM, WINDOW_SIZE_MAX * FRAME_DIM);
        //    accB.resize(WINDOW_SIZE_MAX*FRAME_DIM,WINDOW_SIZE_MAX*FRAME_DIM);
        BMargedP.resize(WINDOW_SIZE_MAX * FRAME_DIM, WINDOW_SIZE_MAX * FRAME_DIM);
        BMargedP.setZero();
        Bupdate.resizeLike(b);
        Bupdate.setZero();
        b.diagonal() = Eigen::Matrix<SCALAR, WINDOW_SIZE_MAX * FRAME_DIM, 1>::Ones()*99999  ;
        //    accB.setIdentity();
        //    BMargedP.setZero();
        for (int i = 0; i < WINDOW_SIZE_MAX; ++i) {
            bIsIdentity[i] = 1;
        }
    }
    void resetForStep() override {
//        BMargedP.setZero();
        Bupdate.setZero();
//        b.setZero();
//        b.diagonal() = Eigen::Matrix<SCALAR, WINDOW_SIZE_MAX * FRAME_DIM, 1>::Ones();
//        //    accB.setIdentity();
//        //    BMargedP.setZero();
//        for (int i = 0; i < WINDOW_SIZE_MAX; ++i) {
//            bIsIdentity[i] = 1;
//        }
    }
    void initapplyRes(ResidualBase_t *residual) override {
        auto *res = static_cast<Residual_t *>(residual);
        Camera_t *host = res->getHost();
        Camera_t *target = res->getTarget();
        //B !!```
        int sth, stt;
        sth = FRAME_DIM * host->getid();
        stt = FRAME_DIM * target->getid();

        Mat accBTH = res->jdrdxi_th.transpose() * res->jdrdxi_th;

        if (ifCamUninit(host->getid())) {
            //        optimizor.setZeroaccBpart(sth,sth,FRAME_DIM,FRAME_DIM);
            setZeroBpart(sth, sth, FRAME_DIM, FRAME_DIM);
            setCaminited(host->getid());
        }
        if (ifCamUninit(target->getid())) {
            setZeroBpart(stt, stt, FRAME_DIM, FRAME_DIM);
            setCaminited(target->getid());
        }
        //    optimizor.updateaccBpart(sth,sth,FRAME_DIM,FRAME_DIM,accBTH);

        b.block(sth, sth, FRAME_DIM, FRAME_DIM)+=res->getAdjH().transpose() * accBTH * res->getAdjH();

        b.block(stt, stt, FRAME_DIM, FRAME_DIM)+=accBTH;

        b.block(sth, stt, FRAME_DIM, FRAME_DIM)+= res->getAdjH().transpose() * accBTH;

        b.block(stt, sth, FRAME_DIM, FRAME_DIM)+= accBTH * res->getAdjH();
    }
    void clearMargedInfo() override {
        BMargedP.setZero();
    }

    void margOnePoint(Point_t *point, int camSize) override {
        for (int k = 0; k < camSize; ++k) {
            for (int l = 0; l < camSize; ++l) {
                int str,stc;
                str = k*FRAME_DIM;
                stc = l*FRAME_DIM;
    //            BMargedP.block<FRAME_DIM,FRAME_DIM>(str,stc)+=
                BMargedP.block(str,stc,FRAME_DIM,FRAME_DIM)+=
                        point->getEik(k)*point->getC().inverse()*point->getEik(l).transpose();
            }
        }
    }

    void stepApplyPoint(Point_t *point, int camSize) override {
        //对每个点，统计信息到B
        //因为残差一变，就得重算Eik，根据总的Eik再重算Bundate
        //只能分成两步，先算Eik，在addresidual里
        //再算bupdate，在这里
        for (int k = 0; k < camSize; ++k) {
            for (int l = 0; l < camSize; ++l) {
                int str, stc;
                str = k * FRAME_DIM;
                stc = l * FRAME_DIM;
                Bupdate.block(str, stc, FRAME_DIM, FRAME_DIM) +=
                        point->getEik(k) * point->getC().inverse() * point->getEik(l).transpose();
            }
        }
    }
    virtual void solveStep(Mat &right, Mat &delta) override {
        //注意，marg掉的信息是附加在bupdate前的,bupdate其实就是EC^{-1}E^T，
        // marg信息其实就是提前将被marg掉的点的信息存下来，所以marg和bupdate是相加的关系
        delta = (b-(Bupdate + BMargedP)).colPivHouseholderQr().solve(right);
    };

    Mat getLeft(){
        return (b);
    }
};

class OptimizerDebugC{
public:
    Mat generateCamUpdate();
    Mat generateTotalH();
    Mat generateTotalV();
    Mat generateTotalJ();
    Mat generateTotalR();
    Mat generateVCam();
    Mat generateVPoint();
    void reset();
    const static int RES_DIM = 1;
    const static int FRAME_DIM = 8;
    const static int WINDOW_SIZE_MAX = 8;
    const static int POINT_DIM = 1;
    typedef double SCALAR;
    using WindowOptimizor_t =  WindowOptimizor<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> ;
    using Hession_t =  Hession<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR> ;
    using Camera_t = WindowOptimizor_t::Camera_t;
    using Point_t = WindowOptimizor_t::Point_t;
    using ResidualBase_t = ResidualBase<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>;
    using Residual_t = Residual<RES_DIM,FRAME_DIM,WINDOW_SIZE_MAX,POINT_DIM,SCALAR>;
    WindowOptimizor_t *optimizor;
    Hession_t *hession;
    void init(int camNum, int resNum);
    void newResidual(Camera_t *host, Camera_t *target, float possibilityCreateNewPoint = 0.01);
    const Mat get_B(){return hession->b.block(0,0,optimizor->getCameras().size()*FRAME_DIM,optimizor->getCameras().size()*FRAME_DIM);}
    vector<Point_t*> getAllPoints();
    vector<ResidualBase_t*> getAllResiduals();
    Mat genBUpdate();
    Mat genE_CInv_VPoint(const Mat &vPoint);
    void printInfo();
    void fetchInfo(int *camNum = nullptr,int *pointNum = nullptr, int *resNum = nullptr);
    OptimizerDebugC (){
        hession = new Hession_t();
        optimizor = new WindowOptimizor_t(hession);
    }
};
void test_init(OptimizerDebugC &system);

void test_add_camera(OptimizerDebugC &system);

void test_marg_point(OptimizerDebugC &system, float possibility);

#endif //WINDOWOPTIMIZATION_OPTIMIZERDEBUGC_H

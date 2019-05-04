//
// Created by libaoyu on 19-4-19.
//

#ifndef WINDOWOPTIMIZATION_RESIDUALBA_H
#define WINDOWOPTIMIZATION_RESIDUALBA_H

#include "types.h"
template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
class ResidualBA:public Residual_t{
public:
    using Base_t = ResidualBase<RES_DIM, FRAME_DIM, WINDOW_SIZE_MAX, POINT_DIM, SCALAR>;
    using WindowOptimizor_t = typename Base_t::WindowOptimizor_t;
    using Camera_t = typename Base_t::Camera_t;
    using Point_t = typename Base_t::Point_t;
    using VectorFrameDim_t = typename Base_t ::VectorFrameDim;
    using VectorPointDim = typename Base_t ::VectorPointDim;
    using ResData = typename Base_t ::ResData;
    using Point3D = typename Point_t::PointState;
    using Point2D = Eigen::Matrix<SCALAR,2,1>;
    using JdpdP_t = Eigen::Matrix<SCALAR,2,POINT_DIM>;
    using JdrdP_t = Eigen::Matrix<SCALAR,RES_DIM,POINT_DIM>;
    using JdPdxi_t = Eigen::Matrix<SCALAR,POINT_DIM,6>;
    using Jdrdp_t = Eigen::Matrix<SCALAR,RES_DIM,2>;
    using Jdrdcam_t = Eigen::Matrix<SCALAR,RES_DIM,FRAME_DIM>;
private:
    Camera_t *host;
    Point_t *point;
    JdrdP_t jdrdPw;
    Jdrdcam_t jdrdcam; //对相机整体
    ResData resdata;


    JdpdP_t jdpcdPc;
    JdrdP_t jdrdPc;
    JdPdxi_t jdPcdxi;
    VectorFrameDim_t jcam_r;
    VectorPointDim jPw_r;
    ResData observasion;
    //运算过程中的变量
    Point3D Pc;
    Point2D pc;
    Jdrdp_t jdrdpc;
public:
    ResidualBA(Camera_t *_host,Point_t *_point,const ResData &_resdata, const ResData &_observasion):
            host(_host),point(_point),
            jdrdPw(DataGenerator::gen_data<JdrdP_t>()),
            jdrdcam(DataGenerator::gen_data<Jdrdcam_t>()),
            observasion(_observasion)
    {
        point->getResiduals().push_back(this);
    }
    ResidualBA(Camera_t *_host,Point_t *_point,const ResData &_observasion):
    ResidualBA(_host,_point,DataGenerator::gen_data<Eigen::Matrix<SCALAR,RES_DIM,1>>(),_observasion){
    }
    const Jdrdxi_t &getJdrdxi()const {return jdrdcam;}
    Camera_t * getHost(){
        return host;
    }
    void computeRes() override {
        // camera[0,1,2，3,4,5] se3
        // 6 focal`
        // 7,8 second and fourth order radial distortion.
//        dbcout<<point->getPointState()<<endl;
        Pc = host->getSE3State() * point->getPointState();
        pc(0) = - Pc[0]/Pc[2];
        pc(1) = - Pc[1] / Pc[2];
        const typename Camera_t::CamState &camState = host->getState();

        const double& l1 = camState[7];
        const double& l2 = camState[8];
        double r2 = pc.transpose()*pc;
        double distortion = 1.0 + r2  * (l1 + l2  * r2);
        const double& focal = camState[6];

        Eigen::Vector2d pfinal = focal*distortion*pc;
        // The error is the difference between the predicted and observed position.
        resdata = pfinal - observasion;


        double A = r2;
        double B = (l1 + A * l2);
        jdrdpc = focal*(
                2.*(B + A*l2)*pc*pc.transpose()
                + distortion*Eigen::Matrix<double,RES_DIM,RES_DIM>::Identity()
        );
        jdpcdPc<<
               -1./Pc(2),  0,          Pc(0)/(Pc(2)*Pc(2)),
                0,          -1./Pc(2),  Pc(1)/(Pc(2)*Pc(2));
        jdrdPc = jdrdpc * jdpcdPc;
        jdPcdxi.block(0,0,3,3).setIdentity();
        jdPcdxi.block(0,3,3,3)=Sophus::SO3<double>::hat(-Pc);

        jdrdcam.block(0,0,2,6) = jdrdPc*jdPcdxi;

        jdrdcam.block(0,6,2,1) = distortion*pc; //drdf
        jdrdcam.block(0,7,2,1) = focal*r2*pc; //drdl1
        jdrdcam.block(0,8,2,1) = focal*r2*r2*pc; //drdl2


        jdrdPw = jdrdPc*host->getSE3State().rotationMatrix();

        jcam_r = jdrdcam.transpose()*resdata;
        jPw_r = jdrdPw.transpose()*resdata;
//        dbcout<<camState<<std::endl;
//        if(abs(point->getPointState()[0] + 12.056) < 0.001){
//            dbcout<<camState[0]<<std::endl;
//            dbcout<<point->getPointState()<<std::endl;
//        }
//        if(abs(point->getPointState()[0] + 12.056) < 0.001 && abs(camState[0] + 0.016944) < 0.001){
//            dbcout<<pfinal<<std::endl;
//            dbcout<<jdrdcam<<std::endl;
//            dbcout<<jdrdPw<<std::endl;
//            exit(-1);
//        }
    };

    void applyDataToPoint(HessionStruct_t *hessionBase) override {
        point->addEik(host->getid(),jdrdcam.transpose()*jdrdPw);
        point->addC(jdrdPw.transpose()*jdrdPw);             //  jdrdp * r

        point->addJp_r(jPw_r);
        host->jx_r += jcam_r;
    }

    SCALAR asScalar() const override{
        SCALAR ret = resdata.norm();
        return ret*ret;
    }
};
using ResidualBA_t = ResidualBA<res_dim,frame_dim,window_size,point_dim, Scalar>;
#endif //WINDOWOPTIMIZATION_RESIDUALBA_H

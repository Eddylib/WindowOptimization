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
    using Jdrdp_t = typename Base_t ::Jdrdp_t;
    using Jdrdxi_t = typename Base_t ::Jdrdxi_t;
    using VectorFrameDim_t = typename Base_t ::VectorFrameDim;
    using VectorPointDim = typename Base_t ::VectorPointDim;
    using ResData = typename Base_t ::ResData;
    using Point3D = typename Point_t::PointState;
    using Point2D = Eigen::Matrix<SCALAR,2,1>;
private:
    Camera_t *host;
    Point_t *point;
    Jdrdp_t jdrdp;
    Jdrdxi_t jdrdxi; //对位姿的求导
    VectorFrameDim_t jxi_r;
    VectorPointDim jp_r;
    ResData resdata;
    ResData observasion;
    //运算过程中的变量
    Point3D projected;
    Point2D projected2d;
    Point2D undistored;
    SCALAR distortion;
    SCALAR focal;
    SCALAR p2dtp2d;
public:
    ResidualBA(Camera_t *_host,Point_t *_point,const ResData &_resdata, const ResData &_observasion):
            host(_host),point(_point),
            jdrdp(DataGenerator::gen_data<Jdrdp_t>()),
            jdrdxi(DataGenerator::gen_data<Jdrdxi_t>()),
            observasion(_observasion)
    {
        jxi_r = jdrdxi.transpose()*resdata;          //  jdrdxi_th *r
        jp_r = jdrdp.transpose()*resdata;
        //other
        if(host)
            host->jx_r -= jxi_r;
        if(point)
            point->addJp_r(jp_r);
    }
    ResidualBA(Camera_t *_host,Point_t *_point,const ResData &_observasion):ResidualBA(_host,_point,DataGenerator::gen_data<Eigen::Matrix<SCALAR,RES_DIM,1>>(),_observasion){

    }
    const Jdrdxi_t &getJdrdxi()const {return jdrdxi;}
    Camera_t * getHost(){
        return host;
    }
    void computeRes() override {
        // camera[0,1,2，3,4,5] se3
        // 6 focal`
        // 7,8 second and fourth order radial distortion.
//        dbcout<<point->getPointState()<<endl;
        projected = host->se3State * point->getPointState();
        projected2d(0) = - projected[0]/projected[2];
        projected2d(1) = - projected[1] / projected[2];
        const SCALAR& l1 = host->state[7];
        const SCALAR& l2 = host->state[8];
        p2dtp2d = projected2d.transpose()*projected2d;;
        SCALAR r2 = p2dtp2d;
        distortion = 1.0 + r2  * (l1 + l2  * r2);

        focal = host->state[6];
        undistored = focal*distortion*projected2d;
        // The error is the difference between the predicted and observed position.
        resdata = undistored - observasion;
    };
    void computeJ() override {
        Jdrdp_t jdrdp_tmp; //2x3
        Jdrdxi_t jdrdxi_tmp; //2x9 xi(1~6) f(7) l1 (8) l2(9)
        Point2D drdf = distortion*projected2d;
        Point2D drdl1 = focal*p2dtp2d*projected2d;
        Point2D drdl2 = focal*p2dtp2d*p2dtp2d*projected2d;
        const SCALAR& l1 = host->state[7];
        const SCALAR& l2 = host->state[8];

        Eigen::Matrix<SCALAR, 2, 2> drdp2d;
        SCALAR px = projected2d(0);
        SCALAR px2 = px*px;
        SCALAR py = projected2d(1);
        SCALAR py2 = py*py;
        SCALAR pxpy = px*py;
        SCALAR A = 1+p2dtp2d;
        SCALAR B = l1 + l2*p2dtp2d;
        drdp2d(0,0) = 2.*px2*B + A*(2*l2*px2 +B);
        drdp2d(0,1) = 2.*(pxpy+l2);
        drdp2d(1,0) = drdp2d(0,1);
        drdp2d(1,1) = 2*A*l2*py2 + 2*py2*B;
        Eigen::Matrix<SCALAR, RES_DIM, POINT_DIM> dpdPpj;
        dpdPpj.setZero();
        dpdPpj(0,0) = -1./projected(2);
        dpdPpj(0,2) = -projected(0);
        dpdPpj(1,1) = -1./projected(2);
        dpdPpj(1,2) = -projected(1);

        Eigen::Matrix<SCALAR,POINT_DIM,POINT_DIM> dPpjdP = host->se3State.rotationMatrix();
        jdrdp = drdp2d*dpdPpj*dPpjdP;

        Eigen::Matrix<SCALAR,POINT_DIM,2*POINT_DIM> dPpjdxi;
        dPpjdxi.block(0,0,3,3).setIdentity();
        Eigen::Matrix<SCALAR,POINT_DIM,POINT_DIM> right = dPpjdxi.block(0,3,3,3);
        right(0,1) = projected(2);
        right(0,2) = -projected(1);
        right(1,0) = -projected(2);
        right(1,2) = projected(0);
        right(2,0) = projected(1);
        right(2,1) = -projected(0);
        Eigen::Matrix<SCALAR,RES_DIM,2*POINT_DIM> drdxi;
        drdxi = drdp2d*dpdPpj*dPpjdxi;

        jdrdxi.block(0,0,RES_DIM,6) = drdxi;
        jdrdxi.block(0,6,RES_DIM,1) = drdf;
        jdrdxi.block(0,7,RES_DIM,1) = drdl1;
        jdrdxi.block(0,8,RES_DIM,1) = drdl2;
    };
    void initApplyDataToPoint(HessionStruct_t *hessionBase) override {
        // one for target, one for host
        point->addEik(host->getid(),jdrdxi.transpose()*jdrdp);
        point->getResiduals().push_back(this);
        point->addC(jdrdp.transpose()*jdrdp);             //  jdrdp * r
    }
};
using ResidualBA_t = ResidualBA<res_dim,frame_dim,window_size,point_dim, Scalar>;
#endif //WINDOWOPTIMIZATION_RESIDUALBA_H

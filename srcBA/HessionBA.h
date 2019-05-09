//
// Created by libaoyu on 19-4-19.
//

#ifndef WINDOWOPTIMIZATION_HESSIONBA_H
#define WINDOWOPTIMIZATION_HESSIONBA_H

#include "types.h"
#include "ResidualBA.h"

template<int RES_DIM, int FRAME_DIM, int WINDOW_SIZE_MAX, int POINT_DIM, typename SCALAR>
class HessionStructBA:public HessionStruct_t{
public:
    using CameraHessilBlock = Eigen::Matrix<SCALAR,FRAME_DIM,FRAME_DIM>;
    int block_num = 0;
private:
    Mat camera_h;
    vector<bool> if_inited;
    void addCamBlock(int i, int j,const Mat &data){
        if(i==j && !if_inited[i]){
            if_inited[i] = 1;
            camera_h.block(i*FRAME_DIM,j*FRAME_DIM,FRAME_DIM,FRAME_DIM) = data;
        }
        else
            camera_h.block(i*FRAME_DIM,j*FRAME_DIM,FRAME_DIM,FRAME_DIM) += data;
    }
    Mat getBlock(int i,int j){
        return camera_h.block(i*FRAME_DIM,j*FRAME_DIM,FRAME_DIM,FRAME_DIM);
    }
    void setCamBlockIdentity(int i,int j,SCALAR scalar){
        camera_h.block(i*FRAME_DIM,j*FRAME_DIM,FRAME_DIM,FRAME_DIM).setIdentity();
        camera_h.block(i*FRAME_DIM,j*FRAME_DIM,FRAME_DIM,FRAME_DIM)*=scalar;
    }
public:
    HessionStructBA(){
        camera_h.resize(WINDOW_SIZE_MAX*FRAME_DIM,WINDOW_SIZE_MAX*FRAME_DIM);
        if_inited.resize(WINDOW_SIZE_MAX);
        for (int i = 0; i < WINDOW_SIZE_MAX; ++i) {
            getBlock(i,i).setIdentity();
            if_inited[i] = 0;
            setCamBlockIdentity(i,i,99999);
        }
        block_num = 0;
    }
    bool ifCamInit(int idx) const{
        return if_inited[idx];
    };
    // ------------------------------------------------
    // 完整marg点相关的在BA中都暂时无用
    // 清空marginlize相关数据结构
    void clearMargedInfo() override {}
    // 边缘化掉一个点,主要涉及到对marginlize相关数据结构的更新
    void margOnePoint(Point_t *point, int camSize) override {}
    // ------------------------------------------------

    void applyRes(ResidualBase_t *residual) override {
        //applyRes算一部分B，这部分结束后Hession矩阵的相机部分和点部分都是对角块矩阵
        auto *res = static_cast<ResidualBA_t *>(residual);
        Camera_t *host = res->getHost();
        //B !!```
        Mat accBTH = res->getJdrdxi().transpose() * res->getJdrdxi();
        addCamBlock(host->getid(),host->getid(),accBTH);
    }

    // 使用舒尔布进行计算对B的更新量
    void stepApplyPoint(Point_t *point, int camSize) override {
        //对每个点，统计信息到B
        //这个函数进行舒尔布的计算，主要是将点的信息更新到相机的hession块上
        for (int k = 0; k < camSize; ++k) {
            for (int l = 0; l < camSize; ++l) {
                if(point->ifHasEik(k) && point->ifHasEik(l))
                    addCamBlock(k,l,
                            -point->getEik(k)
                            * point->getC().inverse()
                            * point->getEik(l).transpose());
            }
        }
    }
    // 进行一次迭代
    void solveStep(Mat &right, Mat &delta) override {
        delta = (camera_h).colPivHouseholderQr().solve(right);
    }
    // 每次迭代前调用，清除stepApplyPoint相关数据内容
    void resetForStep() override {
        camera_h.setZero();
        for (int i = 0; i < WINDOW_SIZE_MAX; ++i) {
            getBlock(i,i).setIdentity();
            if_inited[i] = 0;
            setCamBlockIdentity(i,i,99999);
        }
        block_num = 0;
    }
};
using HessionStructBA_t = HessionStructBA<res_dim,frame_dim,window_size,point_dim, Scalar>;
#endif //WINDOWOPTIMIZATION_HESSIONBA_H

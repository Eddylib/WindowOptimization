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
//        if(camera_h[i][j]){
//            *(camera_h[i][j]) += data;
//        }else{
//            block_num ++;
//            camera_h[i][j] = new CameraHessilBlock();
//            *(camera_h[i][j]) = data;
//        }
        if(i==j && !if_inited[i]){
            if_inited[i] = 1;
            getBlock(i,j) = data;
        }
        else
            getBlock(i,j) += data;
    }
    Mat getBlock(int i,int j){
        return camera_h.block(i*FRAME_DIM,j*FRAME_DIM,FRAME_DIM,FRAME_DIM);
    }
public:
    HessionStructBA(){
        camera_h.resize(WINDOW_SIZE_MAX*FRAME_DIM,WINDOW_SIZE_MAX*FRAME_DIM);
        if_inited.resize(WINDOW_SIZE_MAX);
        for (int i = 0; i < WINDOW_SIZE_MAX; ++i) {
            getBlock(i,i).setIdentity();
            if_inited[i] = 0;
            getBlock(i,i).diagonal() = Eigen::Matrix<SCALAR,FRAME_DIM,1>::Ones()*99999;
        }
        block_num = 0;
    }
    bool ifCamInit(int idx) const{
        return if_inited[idx];
    }
    // ------------------------------------------------
    // 完整marg点相关的在BA中都暂时无用
    // 清空marginlize相关数据结构
    void clearMargedInfo() override {}
    // 边缘化掉一个点,主要涉及到对marginlize相关数据结构的更新
    void margOnePoint(Point_t *point, int camSize) override {}
    // ------------------------------------------------

    void initapplyRes(ResidualBase_t *residual) override {
        auto *res = static_cast<ResidualBA_t *>(residual);
        Camera_t *host = res->getHost();
        //B !!```
        Mat accBTH = res->getJdrdxi().transpose() * res->getJdrdxi();
        addCamBlock(host->getid(),host->getid(),-accBTH);
    }

    // 使用舒尔布进行计算对B的更新量
    void stepApplyPoint(Point_t *point, int camSize) override {

        //对每个点，统计信息到B
        //因为残差一变，就得重算Eik，根据总的Eik再重算Bundate
        //只能分成两步，先算Eik，在addresidual里
        //再算bupdate，在这里
        for (int k = 0; k < camSize; ++k) {
            for (int l = 0; l < camSize; ++l) {
                if(point->ifHasEik(k) && point->ifHasEik(l))
                    addCamBlock(k,l,
                            point->getEik(k)
                            * point->getC().inverse()
                            * point->getEik(l).transpose());
//                Bupdate.block(str, stc, FRAME_DIM, FRAME_DIM) +=
//                        point->getEik(k) * point->getC().inverse() * point->getEik(l).transpose();
            }
        }
    }
    // 进行一次迭代
    virtual void solveStep(Mat &right, Mat &delta) override {}
    // 每次迭代前调用，清除stepApplyPoint相关数据内容
    virtual void resetForStep() override {

    }
};
using HessionStructBA_t = HessionStructBA<res_dim,frame_dim,window_size,point_dim, Scalar>;
#endif //WINDOWOPTIMIZATION_HESSIONBA_H

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
#ifndef WINDOWOPTIMIZATION_WINDOW_OPTIMIZATION_H
#define WINDOWOPTIMIZATION_WINDOW_OPTIMIZATION_H

#include <cassert>
#include <vector>
#include <iostream>

using namespace std;
//#define DEBUG

#include "../alg_config.h"
#include "../alg_utils.h"
#include "../debug_utils.h"
class Camera;
class Point;
class Residual;

class WindowOptimizor{
protected:
    Hxixi B;
    Hxixi accB;// 位姿部分的hession矩阵维护在B中，此矩阵是jdrdx_{ij}^T*jdrdx_{ij}
    //上面两个矩阵的关系是一个仅对相对位姿求导（accB，对应于DSO中的EF什么什么中的acc）
    // 一个是绝对位姿（B对应于DSO中的EF什么什么中的Hm），相对位姿用于跟踪，绝对位姿用于建图

    //由于点被marg而导致的信息增值
    Hxixi BMargedP;
    vector<Camera *> cameras;
    int bIsIdentity[WINDOW_SIZE_MAX];
    void beginMargPoint();
    void endMarg();
    void marginlizeOnePoint(Point *point);
public:
    WindowOptimizor();
    void attach_B();    //B 只有在第一次运算时才会调用此函数将所有数据加载到B上，之后只需进行优化来更新B
    vector<Camera *> & getCameras(){return cameras;}
    void insertCamera(Camera *camera);
    void insertPoint(Point *point);
    void step_once(
            Mat &delta,
            bool if_process_camera = true,
            bool if_process_points = false,
            bool if_update_camera = false,
            bool if_update_points = false);
    //marg掉一个点如何才算成功？ marg前后算位姿数据不变，这才说明信息被合理地添加到了相应的地方。
    void addResidual(Camera *_host, Camera *_target, Point *_point);
    // 外部代码只需要标记某个point为可marg，然后调用marginlizeFlaggedPoint即可marg掉点
    //之后需要调用clearMargedPoints来彻底删除marg掉的点。
    void marginlizeFlaggedPoint();
    void clearMargedPoints();
    void marginlizeOneCamera(Camera *camera);
};
#endif //WINDOWOPTIMIZATION_WINDOW_OPTIMIZATION_H

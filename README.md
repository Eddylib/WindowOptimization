# WindowOptimization

Basic WindowOptimization algorithm in SLAM.

- [x] 各个类数据放到了类本身的结构体里
- [x] 加入EuRoC MAV数据集加载代码，便于以后进一步完善及测试
- [x] 模板化，参数可根据实际情况调整
- [x] Hession矩阵、Residual结构OO化，并修改OptimizerDebug为一个实现样例
- [x] 完成BA中的残差计算、梯度计算
- [x] 迭代过程完成
- [x] 地图点部分的update求解（高优先级，实现完成即是BA主体框架完成）
- [x] 整个过程总结成文章，见[BA的简单实现][1]

todo..
- [ ] 实现完整地marg掉帧，目前项目转为BA求解，这个目前不再进行了
- [ ] LM算法
- [ ] SSE
- [ ] 前端
- [ ] IMU

# 部署
修改CMakeLists.txt中关于Eigen库的位置的变量，即可编译
```
set(EIGEN_DIR path_to_your_own_eigen_dir)
```
修改CMakeLists.txt中关于是否绘图的变量来确定程序是否发布ros消息进行可视化，注意此选项开启后必须安装文章中提到的ros端的包，解压到ros workingspace下的src下，命名为showpath，并在ros中编译。
同时修改ROS_WS确定ros的工作目录。
```
set(ROS_DROW FALSE)
set(ROS_WS /home/libaoyu/Data/catkin_ws)
```


  [1]: http://eddylib.me/archives/24.html






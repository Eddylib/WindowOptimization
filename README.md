# WindowOptimization

Basic WindowOptimization algorithm in SLAM.

- [x] 参照DSO，实现了最基本的相机位姿的marglize方法求解优化步长
- [x] 各个类数据放到了类本身的结构体里
- [x] 实现完整地marg掉点
- [x] 加入EuRoC MAV数据集加载代码，便于以后进一步完善及测试
- [x] 模板化，参数可根据实际情况调整
- [x] Hession矩阵、Residual结构OO化，并修改OptimizerDebug为一个实现样例


todo..
- [ ] 地图点部分的update求解（easy，低优先级）
- [ ] 实现完整地marg掉帧
- [ ] LM算法
- [ ] SSE
- [ ] 前端
- [ ] IMU






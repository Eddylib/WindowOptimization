# WindowOptimization

Basic WindowOptimization algorithm in SLAM.

- [x] 各个类数据放到了类本身的结构体里
- [x] 加入EuRoC MAV数据集加载代码，便于以后进一步完善及测试
- [x] 模板化，参数可根据实际情况调整
- [x] Hession矩阵、Residual结构OO化，并修改OptimizerDebug为一个实现样例
- [x] 完成BA中的残差计算、梯度计算
- [x] 迭代过程完成
- [x] 地图点部分的update求解（高优先级，实现完成即是BA主体框架完成）

todo..
- [ ] 实现完整地marg掉帧，目前项目转为BA求解，这个目前不再进行了
- [ ] LM算法
- [ ] SSE
- [ ] 前端
- [ ] IMU
- [ ] 近期会把整个过程总结成一篇文章，到时请见[我的博客][1]


  [1]: http://eddylib.me






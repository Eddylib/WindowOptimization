#include <iostream>
#include "src/backend/window_optimization.h"
#include "OptimizerDebugC.h"
#include "utils.h"

using namespace std;

int main() {
    timeval t[2];
    OptimizerDebugC windowOptimizor;
    start(t);
    windowOptimizor.init(3,300);
    stop(t);
    cout<<"init duration: "<<duration(t)<<"ms"<<endl;
    cout<<"simple init, also residual and points added"<<endl;
    test_init(windowOptimizor);
    cout<<"add cameras"<<endl;
    test_add_camera(windowOptimizor);
    test_add_camera(windowOptimizor);
    test_add_camera(windowOptimizor);
    test_add_camera(windowOptimizor);
    test_init(windowOptimizor);
    cout<<"maginlize points with possibility settings"<<endl;
    test_marg_point(windowOptimizor,0.5);
    test_init(windowOptimizor);
    return 0;
}
#include <iostream>
#include "src/window_optimization.h"
#include "utils.h"

using namespace std;

int main() {
    timeval t[2];
    OptimizerDebugC windowOptimizor;
    start(t);
    windowOptimizor.init(3,300);
    stop(t);
    cout<<"init duration: "<<duration(t)<<"ms"<<endl;
    test_init(windowOptimizor);
    test_add_camera(windowOptimizor);
    test_add_camera(windowOptimizor);
    test_add_camera(windowOptimizor);
    test_init(windowOptimizor);
    return 0;
}
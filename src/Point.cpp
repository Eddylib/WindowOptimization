//
// Created by libaoyu on 18-11-7.
//

#include "Point.h"
#include "Camera.h"
#include "Residual.h"
int Point::hasResidualWithTarget(int camId) {
    int ret = 0;
    for (int i = 0; i < residuals.size(); ++i) {
        if(residuals[i]->target->getid() == camId){
            ret = 1;
            break;
        }
    }
    return ret;
}
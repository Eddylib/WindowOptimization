//
// Created by libaoyu on 19-1-3.
//

#include "alg_utils.h"

extern const float _G_gradStereoThershold = 100;

#define PATTERN 0

extern int staticPattern[8][2] ={{0,-2}, {-1,-1}, {1,-1},{-2,0},{0,0},{2,0},{-1,1},{0,2}};	// 8 for SSE efficiency};
void nms_(cv::Mat &grad) {

}

std::string path_join(std::initializer_list<std::string> strlist) {
    using namespace std;
    stringstream ret_path;
    auto iter = strlist.begin();
    for(int i = 0; i < strlist.size();i++,iter++){
        auto &item = *iter;
        ret_path<<item;
        if(item[item.length()-1] != '/'&&i<strlist.size()-1){
            ret_path<<'/';
        }
    }
    return ret_path.str();
}

float fixs16_2_float(int16_t in) {
    return in>0?(float(in>>4)+float(in&0x000f)/16.f):-1.f*(float(abs(in)>>4)+float(in&0x000f)/16.f);
}

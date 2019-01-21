//
// Created by libaoyu on 18-12-21.
//

#include "src/io/ImageLoader.h"
#include <iostream>
#include <opencv/cv.hpp>
#include "src/frontend/DispMatcher.h"
#include "src/debug_utils.h"
using namespace std;
void func1(){

    int16_t int16;
    MAVImageLoader loader = MAVImageLoader("/home/libaoyu/Data/slam_dataset/mav0");
    Frame data(loader.get_frame(0));
    cv::imshow("left",data.img_l);
    cv::imshow("right",data.img_r);
    cout<<data.img_l.size()<<endl;
    DispMatcher matcher = DispMatcher();
    cv::Mat output,outputu8,outputuf;
    matcher.do_disp_match(data,output);
    output.convertTo(outputuf,CV_32FC1);
    cv::normalize(output, outputu8, 0, 255, CV_MINMAX, CV_8U);
    cout<<"output.type "<<type2str(output.type())<<endl;
    cout<<"outputu8.type "<<type2str(outputu8.type())<<endl;
    cv::normalize(output, outputu8, 0, 255, CV_MINMAX, CV_8U);
    cv::imshow("disparity",output);
    cv::waitKey(0);
}
int main(){
//    func1();
    MAVImageLoader loader = MAVImageLoader("/home/libaoyu/Data/slam_dataset/mav0");
    Frame data(loader.get_frame(0));
    DispMatcherMy matcher;
    cv::Mat output,outputf;

    matcher.do_disp_match(data,output);
    output.convertTo(outputf,CV_32F,1.f/16.f);
    for (int ii = 0; ii < output.rows; ++ii) {
        for (int jj = 0; jj < output.cols; ++jj) {

            cout<<outputf.at<float>(ii,jj)<<","<<fixs16_2_float(output.at<int16_t >(ii,jj))<<"|";
        }
        cout<<endl;
    }
    cv::imshow("outputf",outputf/60);
    cv::waitKey(0);
}
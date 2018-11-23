//
// Created by libaoyu on 18-11-6.
//

#ifndef WINDOWOPTIMIZATION_UTILS_H
#define WINDOWOPTIMIZATION_UTILS_H

#include <time.h>
#include <sys/time.h>

inline void start(timeval  *time){
    gettimeofday(time, NULL);
}
inline void stop(timeval  *time){
    gettimeofday(time+1, NULL);
}
inline long long duration(timeval  *time){
    return (time[1].tv_sec - time[0].tv_sec)*1000+(time[1].tv_usec - time[0].tv_usec)/1000;
}
#endif //WINDOWOPTIMIZATION_UTILS_H

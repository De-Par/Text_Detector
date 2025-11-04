#pragma once
#if defined(__APPLE__)
    #include <opencv2/opencv.hpp>
#else
    #include <opencv4/opencv2/opencv.hpp>
#endif
#include <vector>
#include "geometry.h"


void draw_and_dump(cv::Mat &img, const std::vector<Detection> &dets);
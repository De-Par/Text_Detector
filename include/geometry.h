#pragma once

#include <array>
#include <vector>

#if defined(__APPLE__)
    #include <opencv2/opencv.hpp>
#else
    #include <opencv4/opencv2/opencv.hpp>
#endif


struct Detection {
    std::array<cv::Point2f, 4> pts; 
    float score;                    
};

void order_quad(cv::Point2f pts[4]);

float contour_score(const cv::Mat &prob, const std::vector<cv::Point> &contour);

float poly_area(const std::vector<cv::Point2f> &p);

float quad_iou(const std::array<cv::Point2f, 4> &A, const std::array<cv::Point2f, 4> &B);
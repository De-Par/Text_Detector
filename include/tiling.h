#pragma once
#if defined(__APPLE__)
    #include <opencv2/opencv.hpp>
#else
    #include <opencv4/opencv2/opencv.hpp>
#endif
#include <vector>
#include <string>
#include "dbnet.h"
#include "geometry.h"


struct GridSpec
{
    int rows = 1;
    int cols = 1;
};

bool parse_tiles(const std::string &s, GridSpec &g);

std::vector<Detection> infer_tiled_unbound(const cv::Mat &img, DBNet &det, const GridSpec &g,
                                           float overlap_frac, double *ms_sum = nullptr, int omp_threads = 0);

std::vector<Detection> infer_tiled_bound(const cv::Mat &img, DBNet &det, const GridSpec &g,
                                         float overlap_frac, double *ms_sum = nullptr, int omp_threads = 0);
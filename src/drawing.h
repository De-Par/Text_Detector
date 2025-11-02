#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "geometry.h"

void draw_and_dump(cv::Mat &img,
                   const std::vector<Detection> &dets);

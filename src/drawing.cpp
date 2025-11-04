#include <iostream>
#include "drawing.h"


void draw_and_dump(cv::Mat &img, const std::vector<Detection> &dets)
{
    for (const auto &d : dets)
    {
        cv::Point p[4] = {
            cv::Point(cvRound(d.pts[0].x), cvRound(d.pts[0].y)),
            cv::Point(cvRound(d.pts[1].x), cvRound(d.pts[1].y)),
            cv::Point(cvRound(d.pts[2].x), cvRound(d.pts[2].y)),
            cv::Point(cvRound(d.pts[3].x), cvRound(d.pts[3].y)),
        };

        for (int i = 0; i < 4; ++i)
            cv::line(img, p[i], p[(i + 1) % 4], cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        std::cout
            << p[0].x << "," << p[0].y << " "
            << p[1].x << "," << p[1].y << " "
            << p[2].x << "," << p[2].y << " "
            << p[3].x << "," << p[3].y
            << std::endl;
    }
}
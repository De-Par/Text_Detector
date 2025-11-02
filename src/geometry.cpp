#include "geometry.h"
#include <algorithm>
#include <cmath>

void order_quad(cv::Point2f pts[4])
{
    // get tl,tr,br,bl from minAreaRect
    std::vector<cv::Point2f> v(pts, pts + 4);
    std::sort(v.begin(), v.end(),
              [](const cv::Point2f &a, const cv::Point2f &b)
              {
                  return (a.y < b.y) || (a.y == b.y && a.x < b.x);
              });

    cv::Point2f tl = v[0].x < v[1].x ? v[0] : v[1];
    cv::Point2f tr = v[0].x < v[1].x ? v[1] : v[0];
    cv::Point2f bl = v[2].x < v[3].x ? v[2] : v[3];
    cv::Point2f br = v[2].x < v[3].x ? v[3] : v[2];

    pts[0] = tl;
    pts[1] = tr;
    pts[2] = br;
    pts[3] = bl;
}

float contour_score(const cv::Mat &prob,
                    const std::vector<cv::Point> &contour)
{
    cv::Rect bbox = cv::boundingRect(contour) &
                    cv::Rect(0, 0, prob.cols, prob.rows);
    if (bbox.empty())
        return 0.f;

    cv::Mat mask = cv::Mat::zeros(bbox.size(), CV_8U);
    std::vector<std::vector<cv::Point>> cnt(1);
    cnt[0].reserve(contour.size());
    for (const auto &p : contour)
    {
        cnt[0].push_back(p - bbox.tl());
    }

    cv::drawContours(mask, cnt, 0, cv::Scalar(255), cv::FILLED);
    cv::Mat roi = prob(bbox);
    cv::Scalar m = cv::mean(roi, mask);
    return static_cast<float>(m[0]);
}

float poly_area(const std::vector<cv::Point2f> &p)
{
    if (p.size() < 3)
        return 0.f;

    double a = 0.0;
    for (size_t i = 0, j = p.size() - 1; i < p.size(); j = i++)
    {
        a += (double)p[j].x * p[i].y - (double)p[i].x * p[j].y;
    }
    return static_cast<float>(std::abs(a) * 0.5);
}

float quad_iou(const std::array<cv::Point2f, 4> &A,
               const std::array<cv::Point2f, 4> &B)
{
    // with OpenCV intersectConvexConvex
    std::vector<cv::Point2f> a(A.begin(), A.end());
    std::vector<cv::Point2f> b(B.begin(), B.end());
    std::vector<cv::Point2f> inter;

    float inter_area = (float)cv::intersectConvexConvex(a, b, inter, true);
    if (inter_area <= 0.f)
        return 0.f;

    float ua = poly_area(a) + poly_area(b) - inter_area;
    return ua > 0.f ? inter_area / ua : 0.f;
}

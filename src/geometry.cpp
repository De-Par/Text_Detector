#include <algorithm>
#include <cmath>
#include <numeric>

#include "geometry.h"


void order_quad(cv::Point2f pts[4]) {
    // sorting: [top-left, top-right, bottom-right, bottom-left]
    std::sort(pts, pts + 4, [](const cv::Point2f &a, const cv::Point2f &b) { 
        return (a.y < b.y) || (a.y == b.y && a.x < b.x); 
    });

    cv::Point2f top1 = pts[0], top2 = pts[1], bot1 = pts[2], bot2 = pts[3];

    if (top1.x > top2.x) std::swap(top1, top2);
    if (bot1.x < bot2.x) std::swap(bot1, bot2);

    pts[0] = top1;
    pts[1] = top2;
    pts[2] = bot1;
    pts[3] = bot2;
}

float contour_score(const cv::Mat &prob, const std::vector<cv::Point> &contour) {
    if (contour.empty())
        return 0.f;
        
    cv::Rect bbox = cv::boundingRect(contour) & cv::Rect(0, 0, prob.cols, prob.rows);
    if (bbox.empty())
        return 0.f;

    cv::Mat mask = cv::Mat::zeros(bbox.size(), CV_8U);
    std::vector<std::vector<cv::Point>> cnt(1);
    cnt[0].reserve(contour.size());

    for (const auto &p : contour)
        cnt[0].push_back(p - bbox.tl());

    cv::drawContours(mask, cnt, 0, cv::Scalar(255), cv::FILLED);
    cv::Mat roi = prob(bbox);
    cv::Scalar m = cv::mean(roi, mask);

    return static_cast<float>(m[0]);
}

float poly_area(const std::vector<cv::Point2f> &p) {
    if (p.size() < 3)
        return 0.f;

    double a = 0.0;
    for (size_t i = 0, j = p.size() - 1; i < p.size(); j = i++)
        a += (double)p[j].x * p[i].y - (double)p[i].x * p[j].y;

    return static_cast<float>(std::abs(a) * 0.5);
}

// Fast IoU approximation by AABB for speedup (enougth for NMS CPU)
static inline float aabb_iou(const std::array<cv::Point2f, 4> &A, const std::array<cv::Point2f, 4> &B) {
    auto minmax = [](const std::array<cv::Point2f, 4> &q) {
        float minx = q[0].x, miny = q[0].y, maxx = q[0].x, maxy = q[0].y;
        for (int i = 1; i < 4; ++i) {
            minx = std::min(minx, q[i].x);
            miny = std::min(miny, q[i].y);
            maxx = std::max(maxx, q[i].x);
            maxy = std::max(maxy, q[i].y);
        }
        return std::array<float, 4>{minx, miny, maxx, maxy};
    };

    auto a = minmax(A), b = minmax(B);
    float interW = std::max(0.f, std::min(a[2], b[2]) - std::max(a[0], b[0]));
    float interH = std::max(0.f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
    float inter = interW * interH;
    float areaA = (a[2] - a[0]) * (a[3] - a[1]);
    float areaB = (b[2] - b[0]) * (b[3] - b[1]);
    float denom = areaA + areaB - inter;

    return (denom > 1e-6f) ? inter / denom : 0.f;
}

// IoU with OpenCV intersectConvexConvex
float quad_iou(const std::array<cv::Point2f, 4> &A, const std::array<cv::Point2f, 4> &B) {
#if USE_FAST_IOU
    return aabb_iou(A, B);
#else
    std::vector<cv::Point2f> a(A.begin(), A.end());
    std::vector<cv::Point2f> b(B.begin(), B.end());
    std::vector<cv::Point2f> inter;

    float inter_area = (float)cv::intersectConvexConvex(a, b, inter, true);
    if (inter_area <= 0.f)
        return 0.f;

    float ua = poly_area(a) + poly_area(b) - inter_area;

    return ua > 0.f ? inter_area / ua : 0.f;
#endif
}
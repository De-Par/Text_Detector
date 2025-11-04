#include <cmath>
#include <algorithm>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include "tiling.h"


bool parse_tiles(const std::string &s, GridSpec &g)
{
    if (s.empty())
        return false;

    size_t pos = s.find_first_of("xX*");
    if (pos == std::string::npos)
        return false;

    int r = 0, c = 0;
    try {
        r = std::stoi(s.substr(0, pos));
        c = std::stoi(s.substr(pos + 1));
    }
    catch (...) { return false; }

    if (r <= 0 || c <= 0)
        return false;

    g.rows = r;
    g.cols = c;

    return true;
}

// --- tiled unbound ---
std::vector<Detection> infer_tiled_unbound(const cv::Mat &img, DBNet &det, const GridSpec &g,
                                           float overlap_frac, double *ms_sum, int /*omp_threads*/)
{
    const int H = img.rows, W = img.cols;

    std::vector<cv::Rect> tiles;
    tiles.reserve(g.rows * g.cols);

    for (int r = 0; r < g.rows; ++r)
    {
        int y0 = (r * H) / g.rows;
        int y1 = ((r + 1) * H) / g.rows;

        for (int c = 0; c < g.cols; ++c)
        {
            int x0 = (c * W) / g.cols;
            int x1 = ((c + 1) * W) / g.cols;

            int tw = x1 - x0, th = y1 - y0;
            int dx = int(std::round(0.5f * overlap_frac * tw));
            int dy = int(std::round(0.5f * overlap_frac * th));

            int ex0 = std::max(0, x0 - dx);
            int ey0 = std::max(0, y0 - dy);
            int ex1 = std::min(W, x1 + dx);
            int ey1 = std::min(H, y1 + dy);

            tiles.emplace_back(ex0, ey0, ex1 - ex0, ey1 - ey0);
        }
    }

    std::vector<Detection> all;
    double sum_ms = 0.0;

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<Detection> local;
        double local_ms = 0.0;
#pragma omp for schedule(dynamic)
        for (int i = 0; i < (int)tiles.size(); ++i)
        {
            const cv::Rect &roi = tiles[i];
            cv::Mat patch = img(roi);

            double ms = 0.0;
            auto dets = det.infer_unbound(patch, &ms);
            for (auto &d : dets)
            {
                for (int k = 0; k < 4; ++k)
                {
                    d.pts[k].x += roi.x;
                    d.pts[k].y += roi.y;
                }
                local.push_back(d);
            }
            local_ms += ms;
        }
#pragma omp critical
        {
            all.insert(all.end(), local.begin(), local.end());
            sum_ms += local_ms;
        }
    }
#else
    for (const auto &roi : tiles)
    {
        cv::Mat patch = img(roi);
        double ms = 0.0;
        auto dets = det.infer_unbound(patch, &ms);

        for (auto &d : dets)
        {
            for (int k = 0; k < 4; ++k)
            {
                d.pts[k].x += roi.x;
                d.pts[k].y += roi.y;
            }
            all.push_back(d);
        }
        sum_ms += ms;
    }
#endif

    if (ms_sum)
        *ms_sum = sum_ms;

    return all;
}

// --- tiled bound ---
std::vector<Detection> infer_tiled_bound(const cv::Mat &img, DBNet &det, const GridSpec &g,
                                         float overlap_frac, double *ms_sum, int omp_threads)
{
    const int H = img.rows, W = img.cols;

    std::vector<cv::Rect> tiles;
    tiles.reserve(g.rows * g.cols);

    for (int r = 0; r < g.rows; ++r)
    {
        int y0 = (r * H) / g.rows;
        int y1 = ((r + 1) * H) / g.rows;

        for (int c = 0; c < g.cols; ++c)
        {
            int x0 = (c * W) / g.cols;
            int x1 = ((c + 1) * W) / g.cols;

            int tw = x1 - x0, th = y1 - y0;
            int dx = int(std::round(0.5f * overlap_frac * tw));
            int dy = int(std::round(0.5f * overlap_frac * th));

            int ex0 = std::max(0, x0 - dx);
            int ey0 = std::max(0, y0 - dy);
            int ex1 = std::min(W, x1 + dx);
            int ey1 = std::min(H, y1 + dy);

            tiles.emplace_back(ex0, ey0, ex1 - ex0, ey1 - ey0);
        }
    }

#ifdef _OPENMP
    int nthreads = (omp_threads > 0 ? omp_threads : omp_get_max_threads());
#else
    int nthreads = 1;
    (void)omp_threads;
#endif

    det.ensure_pool_size(std::max(1, nthreads));

    std::vector<Detection> all;
    double sum_ms = 0.0;

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<Detection> local;
        double local_ms = 0.0;
        int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic)
        for (int i = 0; i < (int)tiles.size(); ++i)
        {
            const cv::Rect &roi = tiles[i];
            cv::Mat patch = img(roi);

            double ms = 0.0;
            auto dets = det.infer_bound(patch, tid, &ms);

            for (auto &d : dets)
            {
                for (int k = 0; k < 4; ++k)
                {
                    d.pts[k].x += roi.x;
                    d.pts[k].y += roi.y;
                }
                local.push_back(d);
            }
            local_ms += ms;
        }
#pragma omp critical
        {
            all.insert(all.end(), local.begin(), local.end());
            sum_ms += local_ms;
        }
    }
#else
    {
        double ms = 0.0;
        auto dets = det.infer_bound(img, 0, &ms);
        all.insert(all.end(), dets.begin(), dets.end());
        sum_ms += ms;
    }
#endif

    if (ms_sum)
        *ms_sum = sum_ms;

    return all;
}
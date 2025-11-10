#include <algorithm>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "tiling.h"


bool parse_tiles(const std::string &s, GridSpec &g) {
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

    if (r < 1 || c < 1)
        return false;

    g.rows = r;
    g.cols = c;

    return (g.rows > 1 || g.cols > 1);
}

static std::vector<cv::Rect> make_tiles(const cv::Size &S, const GridSpec &g, float overlap) {
    int W = S.width, H = S.height;
    float stepX = W / (float)g.cols, stepY = H / (float)g.rows;
    float ovX = std::min(stepX * overlap, stepX * 0.49f);
    float ovY = std::min(stepY * overlap, stepY * 0.49f);

    std::vector<cv::Rect> R;
    R.reserve(g.rows * g.cols);
    for (int r = 0; r < g.rows; ++r) {
        for (int c = 0; c < g.cols; ++c) {
            float x0 = std::max(0.f, c * stepX - ovX);
            float y0 = std::max(0.f, r * stepY - ovY);
            float x1 = std::min((float)W, (c + 1) * stepX + ovX);
            float y1 = std::min((float)H, (r + 1) * stepY + ovY);
            cv::Rect rc((int)std::round(x0), (int)std::round(y0), (int)std::round(x1 - x0), (int)std::round(y1 - y0));
            R.push_back(rc);
        }
    }
    return R;
}

std::vector<Detection> infer_tiled_unbound(const cv::Mat &img, DBNet &det, const GridSpec &g, float overlap_frac, double *ms_sum, int omp_threads) {
    auto tiles = make_tiles(img.size(), g, overlap_frac);
    std::vector<Detection> all;
    all.reserve(128);
    double sum_ms = 0.0;

#ifdef _OPENMP
    (void)omp_threads; // already setup in omp_config

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)tiles.size(); ++i) {
        cv::Mat patch = img(tiles[i]);
        double ms = 0.0;
        auto dets = det.infer_unbound(patch, &ms);

        // move to original coordinates
        for (auto &d : dets) {
            for (int k = 0; k < 4; ++k) {
                d.pts[k].x += tiles[i].x;
                d.pts[k].y += tiles[i].y;
            }
        }

        #pragma omp critical
        {
            sum_ms += ms;
            all.insert(all.end(), dets.begin(), dets.end());
        }
    }
#else
    for (auto &rc : tiles) {
        cv::Mat patch = img(rc);
        double ms = 0.0;
        auto dets = det.infer_unbound(patch, &ms);
        for (auto &d : dets) {
            for (int k = 0; k < 4; ++k) {
                d.pts[k].x += rc.x;
                d.pts[k].y += rc.y;
            }
        }

        sum_ms += ms;
        all.insert(all.end(), dets.begin(), dets.end());
    }
#endif

    if (ms_sum)
        *ms_sum = sum_ms;

    return all;
}

std::vector<Detection> infer_tiled_bound(const cv::Mat &img, DBNet &det, const GridSpec &g, float overlap_frac, double *ms_sum, int omp_threads) {
    auto tiles = make_tiles(img.size(), g, overlap_frac);
    std::vector<Detection> all;
    all.reserve(128);
    double sum_ms = 0.0;

#ifdef _OPENMP
    (void)omp_threads;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)tiles.size(); ++i) {
        int tid = omp_get_thread_num();
        cv::Mat patch = img(tiles[i]);
        double ms = 0.0;
        auto dets = det.infer_bound(patch, tid, &ms);

        for (auto &d : dets) {
            for (int k = 0; k < 4; ++k) {
                d.pts[k].x += tiles[i].x;
                d.pts[k].y += tiles[i].y;
            }
        }

        #pragma omp critical
        {
            sum_ms += ms;
            all.insert(all.end(), dets.begin(), dets.end());
        }
    }
#else
    for (auto &rc : tiles) {
        cv::Mat patch = img(rc);
        double ms = 0.0;
        auto dets = det.infer_bound(patch, 0, &ms);

        for (auto &d : dets) {
            for (int k = 0; k < 4; ++k) {
                d.pts[k].x += rc.x;
                d.pts[k].y += rc.y;
            }
        }

        sum_ms += ms;
        all.insert(all.end(), dets.begin(), dets.end());
    }
#endif

    if (ms_sum)
        *ms_sum = sum_ms;

    return all;
}
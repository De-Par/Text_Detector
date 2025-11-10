#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#if defined(__APPLE__)
    #include <opencv2/imgcodecs.hpp>
#else
    #include <opencv4/opencv2/imgcodecs.hpp>
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "run_bench.h"
#include "tiling.h"
#include "dbnet.h"
#include "nms.h"
#include "timer.h"


// -------- helpers --------
static inline double mean_of(const std::vector<double> &v) {
    if (v.empty())
        return 0.0;
    double s = std::accumulate(v.begin(), v.end(), 0.0);
    return s / double(v.size());
}

static inline double stdev_of(const std::vector<double> &v, double mean) {
    if (v.size() < 2)
        return 0.0;
    long double acc = 0.0L;
    for (double x : v) {
        long double d = x - mean;
        acc += d * d;
    }
    return std::sqrt(double(acc / (v.size() - 1)));
}

static inline double percentile_of(std::vector<double> v, double p) {
    if (v.empty())
        return 0.0;
    if (p <= 0.0)
        return *std::min_element(v.begin(), v.end());
    if (p >= 100.0)
        return *std::max_element(v.begin(), v.end());
    // kth element at floor((p/100)*(n-1))
    const double pos = (p / 100.0) * double(v.size() - 1);
    const size_t k = size_t(std::floor(pos));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

static inline std::string fixed_hw_string(int W, int H) {
    if (W > 0 && H > 0) {
        std::ostringstream os;
        os << W << "x" << H;
        return os.str();
    }
    return std::string("auto");
}

int run_bench(const Options &opt) {
    cv::Mat img = cv::imread(opt.image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Cannot read image: " << opt.image_path << "\n";
        return 2;
    }

    // Tiles?
    GridSpec g{1, 1};
    bool use_tiles = parse_tiles(opt.tiles_arg, g);

    // Decide ORT intra-op threads:
    //   - with tiling (OpenMP outside): ORT=1 to avoid nested parallelism
    //   - without tiling: ORT = user (--threads) or auto (0)
    int ort_intra = use_tiles ? 1 : (opt.ort_threads > 0 ? opt.ort_threads : 0);

#ifdef _OPENMP
    int omp_threads = (opt.tile_omp_threads > 0 ? opt.tile_omp_threads : omp_get_max_threads());
#else
    int omp_threads = 1;
#endif

    DBNet det(opt.model_path, ort_intra, 1);
    det.bin_thresh = opt.bin_thresh;
    det.box_thresh = opt.box_thresh;
    det.unclip_ratio = opt.unclip;
    det.limit_side_len = opt.side;
    det.apply_sigmoid = (opt.apply_sigmoid != 0);

    // For tiles: set fixed HW if not provided to enable stable shapes & memory patterns
    if (use_tiles) {
        if (opt.fixedW > 0 && opt.fixedH > 0) {
            det.fixed_W = opt.fixedW;
            det.fixed_H = opt.fixedH;
        }
        else {
            int tileW = (img.cols + g.cols - 1) / g.cols;
            int tileH = (img.rows + g.rows - 1) / g.rows;
            det.fixed_W = (tileW + 31) & ~31;
            det.fixed_H = (tileH + 31) & ~31;
        }
        if (opt.bind_io)
            det.ensure_pool_size(std::max(1, omp_threads));
    }

    // ----- Config banner -----
    std::cerr << std::fixed << std::setprecision(3);
    std::cerr << "[BENCH][cfg] "
              << "image=" << img.cols << "x" << img.rows
              << " | tiles=" << (use_tiles ? "on" : "off")
              << " grid=" << g.rows << "x" << g.cols
              << " overlap=" << opt.tile_overlap
              << " | ORT_threads=" << ort_intra
              << " | OMP_threads=" << omp_threads
              << " | bind_io=" << (opt.bind_io ? 1 : 0)
              << " | fixed_hw=" << fixed_hw_string(det.fixed_W, det.fixed_H)
              << " | sigmoid=" << (det.apply_sigmoid ? 1 : 0)
              << "\n";

    // ----- Warmup -----
    const int warm_n = std::max(0, opt.warmup);
    std::vector<double> warm;
    warm.reserve(warm_n);

    for (int i = 0; i < warm_n; ++i) {
        double ms = 0.0;
        if (!use_tiles)
            (void)det.infer_bound(img, 0, &ms);
        else
            (void)infer_tiled_bound(img, det, g, opt.tile_overlap, &ms, omp_threads);
        warm.push_back(ms);
    }

    if (!warm.empty()) {
        double w_mean = mean_of(warm);
        double w_med = percentile_of(warm, 50.0);
        double w_p90 = percentile_of(warm, 90.0);
        double w_p95 = percentile_of(warm, 95.0);
        double w_p99 = percentile_of(warm, 99.0);
        double w_min = *std::min_element(warm.begin(), warm.end());
        double w_max = *std::max_element(warm.begin(), warm.end());
        double w_std = stdev_of(warm, w_mean);

        std::cerr << "[BENCH][warmup] "
                  << "n=" << warm.size()
                  << " mean=" << w_mean << " ms"
                  << " median=" << w_med << " ms"
                  << " p90=" << w_p90 << " ms"
                  << " p95=" << w_p95 << " ms"
                  << " p99=" << w_p99 << " ms"
                  << " min=" << w_min << " ms"
                  << " max=" << w_max << " ms"
                  << " std=" << w_std << " ms"
                  << " fps(mean)=" << (w_mean > 0.0 ? 1000.0 / w_mean : 0.0)
                  << "\n";
    }

    // ----- Measure -----
    const int iters = std::max(1, opt.bench_iters);
    std::vector<double> times;
    times.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        double ms = 0.0;
        if (!use_tiles)
            (void)det.infer_bound(img, 0, &ms);
        else
            (void)infer_tiled_bound(img, det, g, opt.tile_overlap, &ms, omp_threads);
        times.push_back(ms);
    }

    // ----- Stats -----
    double avg = mean_of(times);
    double p50 = percentile_of(times, 50.0);
    double p90 = percentile_of(times, 90.0);
    double p95 = percentile_of(times, 95.0);
    double p99 = percentile_of(times, 99.0);
    double tmin = *std::min_element(times.begin(), times.end());
    double tmax = *std::max_element(times.begin(), times.end());
    double stdv = stdev_of(times, avg);

    // human-readable summary
    std::cerr << "[BENCH][run]   "
              << "n=" << times.size()
              << " mean=" << avg << " ms"
              << " median=" << p50 << " ms"
              << " p90=" << p90 << " ms"
              << " p95=" << p95 << " ms"
              << " p99=" << p99 << " ms"
              << " min=" << tmin << " ms"
              << " max=" << tmax << " ms"
              << " std=" << stdv << " ms"
              << " fps(mean)=" << (avg > 0.0 ? 1000.0 / avg : 0.0)
              << "\n";

    // machine-parseable RESULT line 
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "RESULT"
              << ",tiles=" << (use_tiles ? 1 : 0)
              << ",grid=" << g.rows << "x" << g.cols
              << ",overlap=" << opt.tile_overlap
              << ",ort=" << ort_intra
              << ",omp=" << omp_threads
              << ",bind_io=" << (opt.bind_io ? 1 : 0)
              << ",fixed=" << fixed_hw_string(det.fixed_W, det.fixed_H)
              << ",warmup=" << warm_n
              << ",iters=" << iters
              << ",mean_ms=" << avg
              << ",p50_ms=" << p50
              << ",p90_ms=" << p90
              << ",p95_ms=" << p95
              << ",p99_ms=" << p99
              << ",min_ms=" << tmin
              << ",max_ms=" << tmax
              << ",std_ms=" << stdv
              << ",fps_mean=" << (avg > 0.0 ? 1000.0 / avg : 0.0)
              << "\n";

    return 0;
}
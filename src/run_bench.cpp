#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#if defined(__APPLE__)
    #include <opencv2/opencv.hpp>
#else
    #include <opencv4/opencv2/opencv.hpp>
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "run_bench.h"
#include "timer.h"
#include "dbnet.h"
#include "tiling.h"
#include "drawing.h"
#include "cli.h"


static double percentile(std::vector<double> v, double p) {
    if (v.empty())
        return 0.0;
    if (p < 0)
        p = 0;
    if (p > 1)
        p = 1;
    size_t k = std::min(
        v.size() - 1,
        (size_t)std::floor(p * (v.size() - 1)));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

int run_bench(const Options &opt) {
    // sanity
    if (opt.model_path.empty() || opt.image_path.empty()) {
        std::cerr << "BENCH: need --model and --image\n";
        return 1;
    }

    // load image
    cv::Mat img = cv::imread(opt.image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to load image.\n";
        return 2;
    }

    // init detector
    DBNet det(opt.model_path, opt.ort_threads > 0 ? opt.ort_threads : 0, /*inter_threads=*/1);

    det.bin_thresh = opt.bin_thresh;
    det.box_thresh = opt.box_thresh;
    det.limit_side_len = opt.side;
    det.unclip_ratio = opt.unclip;
    det.apply_sigmoid = (opt.apply_sigmoid != 0);

    if (opt.fixedW > 0 && opt.fixedH > 0) {
        det.fixed_W = opt.fixedW;
        det.fixed_H = opt.fixedH;
    }

    // tile config
    GridSpec g{1, 1};
    bool use_tiles = parse_tiles(opt.tiles_arg, g);

    if (opt.bind_io) {
#ifdef _OPENMP
        int nthreads = (opt.tile_omp_threads > 0 ? opt.tile_omp_threads : omp_get_max_threads());
#else
        int nthreads = 1;
#endif
        det.ensure_pool_size(std::max(1, nthreads));
    }

    // ---------- warmup ----------
    for (int i = 0; i < opt.warmup; i++) {
        if (use_tiles) {
            double ms = 0.0;
            if (opt.bind_io) 
                (void)infer_tiled_bound(img, det, g, opt.tile_overlap, &ms, opt.tile_omp_threads);
            else 
                (void)infer_tiled_unbound(img, det, g, opt.tile_overlap, &ms, opt.tile_omp_threads);
        }
        else {
            double ms = 0.0;
            if (opt.bind_io)
                (void)det.infer_bound(img, 0, &ms);
            else
                (void)det.infer_unbound(img, &ms);
        }
    }

    // ---------- measure iterations ----------
    std::vector<double> totals;
    std::vector<double> infers;
    totals.reserve(opt.bench_iters);
    infers.reserve(opt.bench_iters);

    for (int i = 0; i < opt.bench_iters; i++) {
        Timer T;
        T.tic();
        double infer_ms = 0.0;
        std::vector<Detection> dets;

        if (use_tiles) {
            double ms = 0.0;
            if (opt.bind_io) 
                dets = infer_tiled_bound(img, det, g, opt.tile_overlap, &ms, opt.tile_omp_threads);
            else
                dets = infer_tiled_unbound(img, det, g, opt.tile_overlap, &ms, opt.tile_omp_threads);
            infer_ms = ms; 
        }
        else {
            if (opt.bind_io)
                dets = det.infer_bound(img, 0, &infer_ms);
            else
                dets = det.infer_unbound(img, &infer_ms);
        }

        double total_ms = T.toc_ms();

        if (!opt.no_draw) {
            cv::Mat out = img.clone();
            draw_and_dump(out, dets);
        }

        totals.push_back(total_ms);
        infers.push_back(infer_ms);
    }

    // ---------- stats ----------
    auto t50 = percentile(totals, 0.50);
    auto t90 = percentile(totals, 0.90);
    auto t99 = percentile(totals, 0.99);

    auto i50 = percentile(infers, 0.50);
    auto i90 = percentile(infers, 0.90);
    auto i99 = percentile(infers, 0.99);

    double avg = 0.0;
    for (double x : totals)
        avg += x;
    avg /= std::max<size_t>(1, totals.size());

    double fps_p50 = (t50 > 0 ? 1000.0 / t50 : 0.0);

    std::cerr
        << "[BENCH]"
        << " iters=" << opt.bench_iters
        << " total_ms: avg=" << avg
        << " p50=" << t50
        << " p90=" << t90
        << " p99=" << t99
        << " | infer_ms: p50=" << i50
        << " p90=" << i90
        << " p99=" << i99
        << " | fps@p50=" << fps_p50
        << " | bind_io=" << opt.bind_io
        << " | tiles=" << (use_tiles ? opt.tiles_arg : "1x1")
        << " | ORT_threads="
        << (opt.ort_threads > 0 ? opt.ort_threads : 1)
        << std::endl;

    return 0;
}
#include <iostream>

#if defined(__APPLE__)
    #include <opencv2/imgcodecs.hpp>
#else
    #include <opencv4/opencv2/imgcodecs.hpp>
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "run_single.h"
#include "tiling.h"
#include "dbnet.h"
#include "drawing.h"
#include "nms.h"


int run_single(const Options &opt) {
    cv::Mat img = cv::imread(opt.image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Cannot read image\n";
        return 2;
    }

    // tiles?
    GridSpec g{1, 1};
    bool use_tiles = parse_tiles(opt.tiles_arg, g);

    // Decide ORT intra-op threads:
    //  - with tiling (OpenMP outside): ORT=1 to avoid nested parallelism
    //  - without tiling: ORT = user (--threads) or auto (0)
    int ort_intra = use_tiles ? 1 : (opt.ort_threads > 0 ? opt.ort_threads : 0);

    DBNet det(opt.model_path, ort_intra, 1);
    det.bin_thresh = opt.bin_thresh;
    det.box_thresh = opt.box_thresh;
    det.unclip_ratio = opt.unclip;
    det.limit_side_len = opt.side;
    det.apply_sigmoid = (opt.apply_sigmoid != 0);

    // For tiles: set fixed HW if not provided (enables stable shapes & memory patterns)
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
        if (opt.bind_io) {
#ifdef _OPENMP
            int nthreads = (opt.tile_omp_threads > 0 ? opt.tile_omp_threads : omp_get_max_threads());
#else
            int nthreads = 1;
#endif
            det.ensure_pool_size(std::max(1, nthreads));
        }
    }

    std::vector<Detection> dets;
    double tiles_ms = 0.0;

    if (!use_tiles) {
        double ms = 0.0;
        dets = opt.bind_io ? det.infer_bound(img, 0, &ms) : det.infer_unbound(img, &ms);
        tiles_ms = ms;
    }
    else {
        dets = opt.bind_io ? infer_tiled_bound(img, det, g, opt.tile_overlap, &tiles_ms, opt.tile_omp_threads)
                           : infer_tiled_unbound(img, det, g, opt.tile_overlap, &tiles_ms, opt.tile_omp_threads);
        // final NMS across all tiles
        dets = nms_poly(dets, opt.nms_iou);
    }

    std::cerr << "Time: " << tiles_ms << " ms\n";
    if (!opt.no_draw) {
        cv::Mat vis = img.clone();
        draw_and_dump(vis, dets);
        cv::imwrite(opt.out_path, vis);
    }
    else {
        cv::Mat dummy = img;
        draw_and_dump(dummy, dets);
    }
    return 0;
}
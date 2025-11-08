#include <iostream>
#include <vector>
#include <algorithm>

#if defined(__APPLE__)
    #include <opencv2/opencv.hpp>
#else
    #include <opencv4/opencv2/opencv.hpp>
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "run_single.h"
#include "timer.h"
#include "dbnet.h"
#include "tiling.h"
#include "nms.h"
#include "drawing.h"


int run_single(const Options &opt) {
    if (opt.model_path.empty() || opt.image_path.empty()) {
        std::cerr << "Need --model and --image\n";
        return 1;
    }

    try {
        // Read image
        cv::Mat img = cv::imread(opt.image_path, cv::IMREAD_COLOR);
        if (img.empty())
            throw std::runtime_error("Failed to load image: " + opt.image_path);
        
        // Init detector
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

        // Tiles?
        GridSpec g{1, 1};
        bool use_tiles = parse_tiles(opt.tiles_arg, g);

        double infer_ms = 0.0;
        std::vector<Detection> dets;
        Timer T;
        T.tic();

        if (use_tiles) {
            if (opt.bind_io) {
#ifdef _OPENMP
                int nthreads = (opt.tile_omp_threads > 0 ? opt.tile_omp_threads : omp_get_max_threads());
#else
                int nthreads = 1;
#endif
                det.ensure_pool_size(std::max(1, nthreads));
                dets = infer_tiled_bound(img, det, g, opt.tile_overlap, &infer_ms, opt.tile_omp_threads);
            }
            else 
                dets = infer_tiled_unbound(img, det, g, opt.tile_overlap, &infer_ms, opt.tile_omp_threads);
            
            dets = nms_poly(dets, opt.nms_iou);
        }
        else {
            if (opt.bind_io) {
                det.ensure_pool_size(1);
                dets = det.infer_bound(img, 0, &infer_ms);
            }
            else
                dets = det.infer_unbound(img, &infer_ms);
        }

        double total_ms = T.toc_ms();

        cv::Mat out = img.clone();
        draw_and_dump(out, dets);

        if (!opt.out_path.empty() && !opt.no_draw)
            cv::imwrite(opt.out_path, out);
        

        std::cerr
            << "[OK]"
            << " dets=" << dets.size()
            << " | infer_ms=" << infer_ms
            << " | total_ms=" << total_ms
#ifdef _OPENMP
            << " | OMP:on"
#else
            << " | OMP:off"
#endif
            << " | ORT_threads="
            << (opt.ort_threads > 0 ? opt.ort_threads : 1)
            << " | bind_io=" << opt.bind_io
            << std::endl;

        return 0;
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}

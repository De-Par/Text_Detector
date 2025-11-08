#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "cli.h"


void usage(const char *prog) {
    std::cout
        << "Usage: " << prog << "\n"
        << "  --model det.onnx --image input.jpg [--out out.png]\n"
        << "  [--bin_thresh 0.3] [--box_thresh 0.6] [--side 960]\n"
        << "  [--threads N] [--unclip 1.5] [--apply_sigmoid 0|1]\n"
        << "  [--tiles RxC] [--tile_overlap 0.10] [--nms_iou 0.3] [--tile_omp N]\n"
        << "  [--omp_places <cores|threads|sockets|{…}>]\n"
        << "  [--omp_bind <close|spread|master|true|false>]\n"
        << "  [--bind_io 0|1] [--fixed_hw WxH]\n"
        << "  [--bench N] [--warmup K] [--no_draw 0|1]\n";
}

bool parse_cli(int argc, char **argv, Options &o) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char *key) {
            return a == key && (i + 1 < argc);
        };

        if (need("--model"))
            o.model_path = argv[++i];
        else if (need("--image"))
            o.image_path = argv[++i];
        else if (need("--out"))
            o.out_path = argv[++i];
        else if (need("--bin_thresh"))
            o.bin_thresh = std::stof(argv[++i]);
        else if (need("--box_thresh"))
            o.box_thresh = std::stof(argv[++i]);
        else if (need("--side"))
            o.side = std::stoi(argv[++i]);
        else if (need("--threads"))
            o.ort_threads = std::stoi(argv[++i]);
        else if (need("--unclip"))
            o.unclip = std::stof(argv[++i]);
        else if (need("--tiles"))
            o.tiles_arg = argv[++i];
        else if (need("--tile_overlap"))
            o.tile_overlap = std::stof(argv[++i]);
        else if (need("--nms_iou"))
            o.nms_iou = std::stof(argv[++i]);
        else if (need("--tile_omp"))
            o.tile_omp_threads = std::stoi(argv[++i]);
        else if (need("--apply_sigmoid"))
            o.apply_sigmoid = std::stoi(argv[++i]);
        else if (need("--omp_places"))
            o.omp_places_cli = argv[++i];
        else if (need("--omp_bind"))
            o.omp_bind_cli = argv[++i];
        else if (need("--bind_io"))
            o.bind_io = std::stoi(argv[++i]);
        else if (need("--fixed_hw"))
            o.fixed_hw = argv[++i];
        else if (need("--bench"))
            o.bench_iters = std::stoi(argv[++i]);
        else if (need("--warmup"))
            o.warmup = std::stoi(argv[++i]);
        else if (need("--no_draw"))
            o.no_draw = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return false;
        }
        else {
            std::cerr << "Unknown arg: " << a << "\n";
            usage(argv[0]);
            return false;
        }
    }

    auto clampf = [](float v, float lo, float hi) {
        return std::max(lo, std::min(hi, v));
    };

    o.bin_thresh = clampf(o.bin_thresh, 0.f, 1.f);
    o.box_thresh = clampf(o.box_thresh, 0.f, 1.f);

    if (o.tile_overlap < 0.f)
        o.tile_overlap = 0.f;
    if (o.tile_overlap > 0.5f)
        o.tile_overlap = 0.5f;
    if (o.nms_iou < 0.f)
        o.nms_iou = 0.f;
    if (o.nms_iou > 1.f)
        o.nms_iou = 1.f;
    if (o.side < 32)
        o.side = 32;

    if (!o.fixed_hw.empty()) {
        size_t pos = o.fixed_hw.find_first_of("xX*");
        if (pos == std::string::npos) {
            std::cerr << "Bad --fixed_hw format. Use WxH.\n";
            return false;
        }
        try {
            o.fixedW = std::stoi(o.fixed_hw.substr(0, pos));
            o.fixedH = std::stoi(o.fixed_hw.substr(pos + 1));
        }
        catch (...) {
            std::cerr << "Bad --fixed_hw numbers.\n";
            return false;
        }
        if (o.fixedW < 32 || o.fixedH < 32) {
            std::cerr << "--fixed_hw too small.\n";
            return false;
        }
        o.fixedW = (o.fixedW + 31) / 32 * 32;
        o.fixedH = (o.fixedH + 31) / 32 * 32;
    }

    return true;
}
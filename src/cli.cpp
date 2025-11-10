#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>

#include "cli.h"


static bool parse_int(const std::string &s, int &v) {
    try { v = std::stoi(s); return true; } catch (...) { return false; }
}

static bool parse_float(const std::string &s, float &v) {
    try { v = std::stof(s); return true; } catch (...) { return false; }
}

void usage(const char *prog) {
    std::cerr << 
    "Usage:\n"
    "  " << prog << " --model det.onnx --image img.jpg [options]\n\n"
    "Options:\n"
    "  --model PATH               ONNX model path (required)\n"
    "  --image PATH               Input image (required)\n"
    "  --out PATH                 Output image (draw boxes), default: out.png\n"
    "  --bin_thresh F             Binarization threshold, default: 0.3\n"
    "  --box_thresh F             Box score threshold, default: 0.6\n"
    "  --unclip F                 Unclip ratio, default: 1.5\n"
    "  --side N                   Limit side (no-tiles), default: 960\n"
    "  --threads N                ORT intra-op threads (no-tiles). 0=auto\n"
    "  --tiles RxC                Enable tiling (e.g., 3x3)\n"
    "  --tile_overlap F           Overlap fraction [0..0.5], default: 0.10\n"
    "  --tile_omp N               OpenMP threads for tiles. 0=auto(physical cores)\n"
    "  --nms_iou F                NMS IoU threshold, default: 0.30\n"
    "  --apply_sigmoid 0|1        Apply sigmoid on output map, default: 0\n"
    "  --bind_io 0|1              Use I/O binding (recommended), default: 0\n"
    "  --fixed_hw WxH              Fix tile input size (e.g., 480x480). Auto if tiles\n"
    "  --omp_places STR           e.g., cores | threads\n"
    "  --omp_bind STR             e.g., close | spread\n"
    "  --bench N                  Benchmark iterations\n"
    "  --warmup N                 Warmup iterations (bench), default: 20\n"
    "  --no_draw                  Do not write output image\n";
}

bool parse_cli(int argc, char **argv, Options &o) {
    if (argc < 3) return false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](std::string &out) -> bool {
            if (i + 1 >= argc) return false;
            out = argv[++i];
            return true;
        };

        if (a == "--model") { std::string v; if (!next(v)) return false; o.model_path = v; }
        else if (a == "--image") { std::string v; if (!next(v)) return false; o.image_path = v; }
        else if (a == "--out") { std::string v; if (!next(v)) return false; o.out_path = v; }
        else if (a == "--bin_thresh") { std::string v; if (!next(v) || !parse_float(v, o.bin_thresh)) return false; }
        else if (a == "--box_thresh") { std::string v; if (!next(v) || !parse_float(v, o.box_thresh)) return false; }
        else if (a == "--unclip") { std::string v; if (!next(v) || !parse_float(v, o.unclip)) return false; }
        else if (a == "--side") { std::string v; if (!next(v) || !parse_int(v, o.side)) return false; }
        else if (a == "--threads") { std::string v; if (!next(v) || !parse_int(v, o.ort_threads)) return false; }
        else if (a == "--tiles") { std::string v; if (!next(v)) return false; o.tiles_arg = v; }
        else if (a == "--tile_overlap") { std::string v; if (!next(v) || !parse_float(v, o.tile_overlap)) return false; }
        else if (a == "--tile_omp") { std::string v; if (!next(v) || !parse_int(v, o.tile_omp_threads)) return false; }
        else if (a == "--nms_iou") { std::string v; if (!next(v) || !parse_float(v, o.nms_iou)) return false; }
        else if (a == "--apply_sigmoid") { std::string v; if (!next(v) || !parse_int(v, o.apply_sigmoid)) return false; }
        else if (a == "--omp_places") { std::string v; if (!next(v)) return false; o.omp_places_cli = v; }
        else if (a == "--omp_bind") { std::string v; if (!next(v)) return false; o.omp_bind_cli = v; }
        else if (a == "--bind_io") { std::string v; if (!next(v) || !parse_int(v, o.bind_io)) return false; }
        else if (a == "--fixed_hw") { std::string v; if (!next(v)) return false; o.fixed_hw = v; }
        else if (a == "--bench") { std::string v; if (!next(v) || !parse_int(v, o.bench_iters)) return false; }
        else if (a == "--warmup") { std::string v; if (!next(v) || !parse_int(v, o.warmup)) return false; }
        else if (a == "--no_draw") { o.no_draw = 1; }
        else { 
            std::cerr << "Unknown arg: " << a << "\n"; 
            return false; 
        }
    }

    if (o.model_path.empty() || o.image_path.empty())
        return false;

    if (!o.fixed_hw.empty()) {
        size_t pos = o.fixed_hw.find_first_of("xX*");
        if (pos == std::string::npos) {
            std::cerr << "Bad --fixed_hw format. Use WxH\n";
            return false;
        }
        try {
            o.fixedW = std::stoi(o.fixed_hw.substr(0, pos));
            o.fixedH = std::stoi(o.fixed_hw.substr(pos + 1));
        }
        catch (...) {
            std::cerr << "Bad --fixed_hw numbers\n";
            return false;
        }
        if (o.fixedW < 32 || o.fixedH < 32) {
            std::cerr << "Parameter values of --fixed_hw are too small (<32)\n";
            return false;
        }
        o.fixedW = (o.fixedW + 31) / 32 * 32;
        o.fixedH = (o.fixedH + 31) / 32 * 32;
    }

    auto clampf = [](float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); };

    o.bin_thresh = clampf(o.bin_thresh, 0.f, 1.f);
    o.box_thresh = clampf(o.box_thresh, 0.f, 1.f);
    o.nms_iou = clampf(o.nms_iou, 0.f, 1.f);
    o.tile_overlap = clampf(o.tile_overlap, 0.f, 0.5f);

    if (o.side < 32)
        o.side = 32;

    return true;
}
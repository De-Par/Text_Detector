#pragma once
#include <string>

struct CliOptions
{
    std::string model_path;
    std::string image_path;
    std::string out_path = "out.png";

    float bin_thresh = 0.3f;
    float box_thresh = 0.6f;
    float unclip = 1.5f;
    int side = 960;
    int ort_threads = 0;

    std::string tiles_arg;
    float tile_overlap = 0.10f;
    float nms_iou = 0.30f;
    int tile_omp_threads = 0;
    int apply_sigmoid = 0;

    std::string omp_places_cli;
    std::string omp_bind_cli;

    int bind_io = 0;
    std::string fixed_hw;
    int fixedW = 0;
    int fixedH = 0;

    int bench_iters = 0;
    int warmup = 20;
    int no_draw = 0;
};

void usage(const char *prog);

bool parse_cli(int argc, char **argv, CliOptions &opt);

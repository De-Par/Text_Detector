#if defined(__APPLE__)
    #include <opencv2/opencv.hpp>
#else
    #include <opencv4/opencv2/opencv.hpp>
#endif
#include "cli.h"
#include "omp_config.h"
#include "run_bench.h"
#include "run_single.h"


int main(int argc, char **argv)
{
    Options opt;
    if (!parse_cli(argc, argv, opt)) 
        return EXIT_FAILURE;
    
    configure_openmp_affinity(opt.omp_places_cli, opt.omp_bind_cli, opt.tile_omp_threads);

    cv::setUseOptimized(true);
    cv::setNumThreads(0);

    if (opt.bench_iters > 0)
        return run_bench(opt);
    else
        return run_single(opt);
}
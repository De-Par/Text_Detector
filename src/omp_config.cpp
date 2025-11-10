#include <thread>
#include <cstdlib>

#if defined(__APPLE__)
    #include <sys/types.h>
    #include <sys/sysctl.h>
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "omp_config.h"


static void set_env_if_empty(const char *k, const char *v) {
    const char *cur = std::getenv(k);
    if (!cur || !*cur) setenv(k, v, 1);
}

void configure_openmp_affinity(const std::string &omp_places_cli, const std::string &omp_bind_cli, int tile_omp_threads) {
    // OMP_PLACES / OMP_PROC_BIND
    if (!omp_places_cli.empty())
        setenv("OMP_PLACES", omp_places_cli.c_str(), 1);
    else
        set_env_if_empty("OMP_PLACES", "cores");

    if (!omp_bind_cli.empty())
        setenv("OMP_PROC_BIND", omp_bind_cli.c_str(), 1);
    else
        set_env_if_empty("OMP_PROC_BIND", "close");

#ifdef _OPENMP
    int n = tile_omp_threads;
    if (n <= 0) {
    #if defined(__APPLE__)
        int phys = 0;
        size_t sz = sizeof(phys);
        if (sysctlbyname("hw.physicalcpu", &phys, &sz, nullptr, 0) == 0 && phys > 0)
            n = phys;
        else {
            unsigned lg = std::thread::hardware_concurrency();
            n = (int)((lg >= 2) ? (lg / 2) : (lg ? lg : 1));
        }
    #else
        unsigned lg = std::thread::hardware_concurrency();
        n = (int)((lg >= 2) ? (lg / 2) : (lg ? lg : 1));
    #endif
    }
    omp_set_dynamic(0);
    omp_set_num_threads(n);
#else
    (void)tile_omp_threads;
#endif
}
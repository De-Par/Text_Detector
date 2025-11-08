#include <cstdlib>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "omp_config.h"


static void inline setenv_if_unset(const char *key, const char *val) {
    if (!std::getenv(key)) 
        setenv(key, val, 1);
}

void configure_openmp_affinity(const std::string &omp_places_cli, const std::string &omp_bind_cli, int tile_omp_threads) {
    // OMP_PLACES
    if (!omp_places_cli.empty())
        setenv("OMP_PLACES", omp_places_cli.c_str(), 1);
    else
        setenv_if_unset("OMP_PLACES", "cores");

    // OMP_PROC_BIND
    if (!omp_bind_cli.empty())
        setenv("OMP_PROC_BIND", omp_bind_cli.c_str(), 1);
    else
        setenv_if_unset("OMP_PROC_BIND", "close");

#ifdef _OPENMP
    if (tile_omp_threads > 0)
        omp_set_num_threads(tile_omp_threads);
#else
    (void)tile_omp_threads;
#endif
}
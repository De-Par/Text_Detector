#pragma once

#include <chrono>


struct Timer {
    std::chrono::steady_clock::time_point t0;

    inline void tic() { t0 = std::chrono::steady_clock::now(); }

    inline double toc_ms() const {
        using namespace std::chrono;
        return duration<double, std::milli>(steady_clock::now() - t0).count();
    }
};
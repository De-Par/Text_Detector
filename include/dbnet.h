#pragma once
#include <string>
#include <vector>
#include <memory>
#if defined(__APPLE__)
    #include <onnxruntime_cxx_api.h>
    #include <opencv2/opencv.hpp>
#else
    #include <onnxruntime/core/session/onnxruntime_cxx_api.h>
    #include <opencv4/opencv2/opencv.hpp>
#endif
#include "timer.h"
#include "geometry.h"


class DBNet
{
public:
    float bin_thresh = 0.3f;
    float box_thresh = 0.6f;
    int limit_side_len = 960;
    float unclip_ratio = 1.5f;
    int min_text_size = 3;
    bool apply_sigmoid = false;

    int fixed_W = 0;
    int fixed_H = 0;

    // ORT objects
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dbnet"};
    Ort::Session session{nullptr};
    Ort::SessionOptions so;

    Ort::AllocatorWithDefaultOptions alloc;
    std::string in_name;
    std::string out_name;

    struct BindingCtx
    {
        Ort::IoBinding io;
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<float> in_buf;
        std::vector<float> out_buf;
        std::vector<int64_t> in_shape;
        std::vector<int64_t> out_shape;

        int curW = 0;
        int curH = 0;
        int curOW = 0;
        int curOH = 0;
        bool bound = false;

        explicit BindingCtx(Ort::Session &s) : io(s) {}
    };

    std::vector<std::unique_ptr<BindingCtx>> pool;

    DBNet(const std::string &model_path, int intra_threads = 0, int inter_threads = 1);

    std::vector<Detection> infer_unbound(const cv::Mat &img_bgr, double *ms_out = nullptr);

    std::vector<Detection> infer_bound(const cv::Mat &img_bgr, 
                                       int ctx_idx, 
                                       double *ms_out = nullptr);

    void ensure_pool_size(int n);

    std::vector<Detection> postprocess(const cv::Mat &prob_map,
                                       float scale_h,
                                       float scale_w,
                                       const cv::Size &orig_size) const;

private:
    void preprocess_dynamic(const cv::Mat &img_bgr,
                            cv::Mat &resized,
                            cv::Mat &blob) const;

    void preprocess_fixed_into(float *dst_chw,
                              const cv::Mat &img_bgr,
                              int W, int H) const;

    std::pair<int, int> probe_out_shape_for(int W, int H);

    void prepare_binding(BindingCtx &ctx, int W, int H);
};
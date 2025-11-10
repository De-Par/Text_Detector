#include <cmath>
#include <stdexcept>

#if defined(__APPLE__)
    #include <opencv2/imgproc.hpp>
    #include <opencv2/imgcodecs.hpp>
#else
    #include <opencv4/opencv2/imgproc.hpp>
    #include <opencv4/opencv2/imgcodecs.hpp>
#endif

#include "dbnet.h"
#include "nms.h"


static inline float sigmoidf(float x) { return 1.f / (1.f + std::exp(-x)); }

DBNet::DBNet(const std::string &model_path, int intra_threads, int inter_threads) {
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    if (intra_threads >= 0)
        so.SetIntraOpNumThreads(intra_threads);
    if (inter_threads >= 0)
        so.SetInterOpNumThreads(inter_threads);

    so.AddConfigEntry("session.intra_op.allow_spinning", "1");
    session = Ort::Session(env, model_path.c_str(), so);

    {
        Ort::AllocatedStringPtr in0 = session.GetInputNameAllocated(0, alloc);
        Ort::AllocatedStringPtr out0 = session.GetOutputNameAllocated(0, alloc);
        in_name = in0.get() ? in0.get() : std::string("input");
        out_name = out0.get() ? out0.get() : std::string("output");
        // memory is owned by AllocatedStringPtr and will be freed automatically
    }
}

void DBNet::ensure_pool_size(int n) {
    if (n <= 0)
        n = 1;
    if ((int)pool.size() >= n)
        return;
    int old = (int)pool.size();
    pool.resize(n);
    for (int i = old; i < n; ++i)
        pool[i] = std::make_unique<BindingCtx>(session);
}

void DBNet::preprocess_dynamic(const cv::Mat &img_bgr, cv::Mat &resized, cv::Mat &blob) const {
    int h = img_bgr.rows, w = img_bgr.cols;
    float scale = 1.0f;
    if (limit_side_len > 0) {
        int max_side = std::max(h, w);
        if (max_side > limit_side_len)
            scale = (float)limit_side_len / (float)max_side;
    }
    int nh = std::max(1, (int)std::round(h * scale));
    int nw = std::max(1, (int)std::round(w * scale));
    nh = (nh + 31) & ~31;
    nw = (nw + 31) & ~31;

    cv::resize(img_bgr, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

    blob.create(1, 3 * nh * nw, CV_32F);
    float *dst = (float *)blob.data;
    const int stride = nh * nw;
    const float s = 1.0f / 255.0f;

    for (int y = 0; y < nh; ++y) {
        const uchar *p = resized.ptr<uchar>(y);
        for (int x = 0; x < nw; ++x) {
            int j = 3 * x;
            float B = p[j + 0] * s, G = p[j + 1] * s, R = p[j + 2] * s;
            int idx = y * nw + x;
            dst[0 * stride + idx] = R;
            dst[1 * stride + idx] = G;
            dst[2 * stride + idx] = B;
        }
    }
}

void DBNet::preprocess_fixed_into(float *dst_chw, const cv::Mat &img_bgr, int W, int H) const {
    cv::Mat resized;
    cv::resize(img_bgr, resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
    const int stride = W * H;
    const float s = 1.0f / 255.0f;

    for (int y = 0; y < H; ++y) {
        const uchar *p = resized.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            int j = 3 * x;
            float B = p[j + 0] * s, G = p[j + 1] * s, R = p[j + 2] * s;
            int idx = y * W + x;
            dst_chw[0 * stride + idx] = R;
            dst_chw[1 * stride + idx] = G;
            dst_chw[2 * stride + idx] = B;
        }
    }
}

std::pair<int, int> DBNet::probe_out_shape_for(int W, int H) {
    std::vector<int64_t> ishape = {1, 3, H, W};
    std::vector<float> dummy(3 * H * W, 0.f);

    Ort::Value it = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
        dummy.data(), dummy.size(), ishape.data(), ishape.size()
    );

    const char *in_names[] = {in_name.c_str()};
    const char *out_names[] = {out_name.c_str()};

    auto out = session.Run(Ort::RunOptions{nullptr}, in_names, &it, 1, out_names, 1);
    if (out.size() != 1 || !out[0].IsTensor())
        throw std::runtime_error("Unexpected output in probe");

    auto ti = out[0].GetTensorTypeAndShapeInfo();
    auto dims = ti.GetShape();
    int oh = 0, ow = 0;

    if (dims.size() == 4) {
        oh = (int)dims[2];
        ow = (int)dims[3];
    }
    else if (dims.size() == 3) {
        oh = (int)dims[1];
        ow = (int)dims[2];
    }
    else if (dims.size() == 2) {
        oh = (int)dims[0];
        ow = (int)dims[1];
    }
    else
        throw std::runtime_error("Unsupported output rank");

    return {oh, ow};
}

void DBNet::prepare_binding(BindingCtx &ctx, int W, int H) {
    if (ctx.bound && ctx.curW == W && ctx.curH == H)
        return;

    ctx.in_shape = {1, 3, H, W};

    auto [oh, ow] = probe_out_shape_for(W, H);
    ctx.out_shape = {1, 1, oh, ow};

    size_t inN = (size_t)1 * 3 * H * W;
    size_t outN = (size_t)1 * 1 * oh * ow;
    ctx.in_buf.resize(inN);
    ctx.out_buf.resize(outN);

    ctx.io.ClearBoundInputs();
    ctx.io.ClearBoundOutputs();

    Ort::Value in = Ort::Value::CreateTensor<float>(ctx.mem, ctx.in_buf.data(), ctx.in_buf.size(), ctx.in_shape.data(), ctx.in_shape.size());
    Ort::Value out = Ort::Value::CreateTensor<float>(ctx.mem, ctx.out_buf.data(), ctx.out_buf.size(), ctx.out_shape.data(), ctx.out_shape.size());

    ctx.io.BindInput(in_name.c_str(), in);
    ctx.io.BindOutput(out_name.c_str(), out);

    ctx.curW = W;
    ctx.curH = H;
    ctx.curOH = oh;
    ctx.curOW = ow;
    ctx.bound = true;
}

std::vector<Detection> DBNet::postprocess(const cv::Mat &prob_map, float /*scale_h*/, float /*scale_w*/, const cv::Size &orig) const {
    // upsample to original size
    cv::Mat prob_up;
    cv::resize(prob_map, prob_up, orig, 0, 0, cv::INTER_LINEAR);

    if (apply_sigmoid) {
        // Safe loop instead of Mat::forEach to avoid UB warning on macOS
        for (int y = 0; y < prob_up.rows; ++y) {
            float *row = prob_up.ptr<float>(y);
            for (int x = 0; x < prob_up.cols; ++x)
                row[x] = sigmoidf(row[x]);
        }
    }

    cv::Mat bin;
    cv::threshold(prob_up, bin, bin_thresh, 255.0, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Detection> dets;
    dets.reserve(contours.size());
    for (auto &c : contours) {
        if (c.size() < 3)
            continue;

        float sc = contour_score(prob_up, c);
        if (sc < box_thresh)
            continue;

        cv::RotatedRect rr = cv::minAreaRect(c);
        rr.size.width *= unclip_ratio;
        rr.size.height *= unclip_ratio;

        cv::Point2f q[4];
        rr.points(q);
        order_quad(q);

        Detection d;
        d.score = sc;
        for (int k = 0; k < 4; ++k)
            d.pts[k] = q[k];

        float w = std::hypot(d.pts[1].x - d.pts[0].x, d.pts[1].y - d.pts[0].y);
        float h = std::hypot(d.pts[3].x - d.pts[0].x, d.pts[3].y - d.pts[0].y);
        if (w < min_text_size || h < min_text_size)
            continue;

        dets.push_back(d);
    }
    return dets;
}

std::vector<Detection> DBNet::infer_unbound(const cv::Mat &img_bgr, double *ms_out) {
    Timer t;
    t.tic();

    cv::Mat resized, blob;
    preprocess_dynamic(img_bgr, resized, blob);

    std::vector<int64_t> ishape = {1, 3, resized.rows, resized.cols};
    Ort::Value in = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
        (float *)blob.data, (size_t)blob.total(),
        ishape.data(), ishape.size()
    );

    const char *in_names[] = {in_name.c_str()};
    const char *out_names[] = {out_name.c_str()};

    auto out = session.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    if (out.size() != 1 || !out[0].IsTensor())
        throw std::runtime_error("Bad output");

    auto ti = out[0].GetTensorTypeAndShapeInfo();
    auto dims = ti.GetShape();
    int oh = 0, ow = 0;

    if (dims.size() == 4) {
        oh = (int)dims[2];
        ow = (int)dims[3];
    }
    else if (dims.size() == 3) {
        oh = (int)dims[1];
        ow = (int)dims[2];
    }
    else if (dims.size() == 2) {
        oh = (int)dims[0];
        ow = (int)dims[1];
    }
    else
        throw std::runtime_error("Unsupported output rank");

    const float *prob = out[0].GetTensorData<float>();
    cv::Mat prob_map(oh, ow, CV_32F, const_cast<float *>(prob));

    auto dets = postprocess(prob_map, (float)oh / resized.rows, (float)ow / resized.cols, img_bgr.size());
    if (ms_out)
        *ms_out = t.toc_ms();

    return nms_poly(dets, 0.30f);
}

std::vector<Detection> DBNet::infer_bound(const cv::Mat &img_bgr, int ctx_idx, double *ms_out) {
    if (pool.empty())
        ensure_pool_size(1);
    if (ctx_idx < 0 || ctx_idx >= (int)pool.size())
        ctx_idx = 0;
        
    auto &ctx = *pool[ctx_idx];

    int W = fixed_W > 0 ? fixed_W : ((limit_side_len > 0) ? limit_side_len : 640);
    int H = fixed_H > 0 ? fixed_H : ((limit_side_len > 0) ? limit_side_len : 640);
    W = (W + 31) & ~31;
    H = (H + 31) & ~31;

    prepare_binding(ctx, W, H);
    preprocess_fixed_into(ctx.in_buf.data(), img_bgr, W, H);

    Timer t;
    t.tic();
    session.Run(Ort::RunOptions{nullptr}, ctx.io);

    cv::Mat prob_map(ctx.curOH, ctx.curOW, CV_32F, ctx.out_buf.data());
    auto dets = postprocess(prob_map, (float)ctx.curOH / H, (float)ctx.curOW / W, img_bgr.size());
    if (ms_out)
        *ms_out = t.toc_ms();

    return dets;
}
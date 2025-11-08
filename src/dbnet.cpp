#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <cmath>

#include "dbnet.h"


DBNet::DBNet(const std::string &model_path, int intra_threads, int inter_threads) {
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (intra_threads > 0)
        so.SetIntraOpNumThreads(intra_threads);
    if (inter_threads > 0)
        so.SetInterOpNumThreads(inter_threads);

    session = Ort::Session(env, model_path.c_str(), so);

    in_name = session.GetInputNameAllocated(0, alloc).get();
    out_name = session.GetOutputNameAllocated(0, alloc).get();
}

// ---------- preprocessing dynamic size ----------
void DBNet::preprocess_dynamic(const cv::Mat &img_bgr, cv::Mat &resized, cv::Mat &blob) const {
    if (img_bgr.empty())
        throw std::runtime_error("Empty input image.");

    int h = img_bgr.rows, w = img_bgr.cols;
    float scale = 1.0f;
    int max_side = std::max(h, w);

    if (max_side > limit_side_len)
        scale = static_cast<float>(limit_side_len) / max_side;

    int nh = std::max(32, static_cast<int>(std::round(h * scale)));
    int nw = std::max(32, static_cast<int>(std::round(w * scale)));

    nh = (nh + 31) / 32 * 32;
    nw = (nw + 31) / 32 * 32;

    cv::resize(img_bgr, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    const cv::Scalar mean(0.485, 0.456, 0.406);
    const cv::Scalar stdev(0.229, 0.224, 0.225);
    cv::Mat norm = (rgb - mean) / stdev;

    std::vector<cv::Mat> ch(3);
    cv::split(norm, ch);

    const int hw = resized.rows * resized.cols;
    blob = cv::Mat(1, 3 * hw, CV_32F);

    float *dst = blob.ptr<float>();
    std::memcpy(dst + 0 * hw, ch[0].reshape(1, 1).ptr<float>(), hw * sizeof(float));
    std::memcpy(dst + 1 * hw, ch[1].reshape(1, 1).ptr<float>(), hw * sizeof(float));
    std::memcpy(dst + 2 * hw, ch[2].reshape(1, 1).ptr<float>(), hw * sizeof(float));
}

// ---------- preprocessing fixed size ----------
void DBNet::preprocess_fixed_into(float *dst_chw, const cv::Mat &img_bgr, int W, int H) const {
    cv::Mat resized;
    cv::resize(img_bgr, resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    const cv::Scalar mean(0.485, 0.456, 0.406);
    const cv::Scalar stdev(0.229, 0.224, 0.225);
    cv::Mat norm = (rgb - mean) / stdev;

    std::vector<cv::Mat> ch(3);
    cv::split(norm, ch);

    const int hw = H * W;
    std::memcpy(dst_chw + 0 * hw, ch[0].reshape(1, 1).ptr<float>(), hw * sizeof(float));
    std::memcpy(dst_chw + 1 * hw, ch[1].reshape(1, 1).ptr<float>(), hw * sizeof(float));
    std::memcpy(dst_chw + 2 * hw, ch[2].reshape(1, 1).ptr<float>(), hw * sizeof(float));
}

// ---------- postprocessing ----------
std::vector<Detection> DBNet::postprocess(const cv::Mat &prob_map, float scale_h, float scale_w, const cv::Size &orig_size) const {
    cv::Mat bin;
    cv::threshold(prob_map, bin, bin_thresh, 1.0, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U, 255.0);

    int k = std::max(1, int(std::round(unclip_ratio)));
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * k + 1, 2 * k + 1));
    cv::dilate(bin, bin, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Detection> dets;
    dets.reserve(contours.size());

    for (auto &c : contours) {
        if (c.size() < 3)
            continue;

        float score = contour_score(prob_map, c);
        if (score < box_thresh)
            continue;

        cv::RotatedRect rr = cv::minAreaRect(c);
        if (std::min(rr.size.width, rr.size.height) < min_text_size)
            continue;

        cv::Point2f pts[4];
        rr.points(pts);
        order_quad(pts);

        Detection d{};
        for (int i = 0; i < 4; ++i) {
            pts[i].x = pts[i].x / scale_w;
            pts[i].y = pts[i].y / scale_h;
            pts[i].x = std::clamp(pts[i].x, 0.0f, float(orig_size.width - 1));
            pts[i].y = std::clamp(pts[i].y, 0.0f, float(orig_size.height - 1));
            d.pts[i] = pts[i];
        }
        d.score = score;
        dets.push_back(d);
    }
    return dets;
}

// ---------- simple Run() without IOBinding ----------
std::vector<Detection> DBNet::infer_unbound(const cv::Mat &img_bgr, double *ms_out) {
    Timer T;
    T.tic();

    cv::Mat resized, blob;
    if (fixed_W > 0 && fixed_H > 0) {
        const int hw = fixed_W * fixed_H;
        blob = cv::Mat(1, 3 * hw, CV_32F);
        preprocess_fixed_into(blob.ptr<float>(), img_bgr, fixed_W, fixed_H);
        resized.create(fixed_H, fixed_W, CV_8UC3);
    }
    else
        preprocess_dynamic(img_bgr, resized, blob);
    

    const int nh = (fixed_H > 0 ? fixed_H : resized.rows);
    const int nw = (fixed_W > 0 ? fixed_W : resized.cols);

    std::vector<int64_t> input_shape = {1, 3, nh, nw};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem,
        blob.ptr<float>(),
        blob.total(),
        input_shape.data(),
        input_shape.size()
    );

    const char *input_names[] = {in_name.c_str()};
    const char *output_names[] = {out_name.c_str()};

    auto out = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    if (out.size() != 1)
        throw std::runtime_error("Unexpected number of outputs (expected 1).");

    float *out_data = out[0].GetTensorMutableData<float>();
    auto out_shape = out[0].GetTensorTypeAndShapeInfo().GetShape();

    int oh = 0, ow = 0;
    bool ok = false;

    // NCHW [1,1,H,W]
    if (out_shape.size() == 4 && out_shape[0] == 1 && out_shape[1] == 1) {
        oh = (int)out_shape[2];
        ow = (int)out_shape[3];
        ok = true;
    }
    // NHWC [1,H,W,1]
    else if (out_shape.size() == 4 && out_shape[0] == 1 && out_shape[3] == 1) {
        oh = (int)out_shape[1];
        ow = (int)out_shape[2];
        ok = true;
    }
    // [1,H,W]
    else if (out_shape.size() == 3 && out_shape[0] == 1) {
        oh = (int)out_shape[1];
        ow = (int)out_shape[2];
        ok = true;
    }
    // [H,W]
    else if (out_shape.size() == 2) {
        oh = (int)out_shape[0];
        ow = (int)out_shape[1];
        ok = true;
    }

    if (!ok || oh <= 0 || ow <= 0)
        throw std::runtime_error("Unexpected output shape.");

    cv::Mat prob_map(oh, ow, CV_32F, out_data);
    cv::Mat prob = prob_map.clone();

    if (apply_sigmoid) {
        cv::Mat ex;
        cv::exp(-prob, ex);
        prob = 1.0f / (1.0f + ex);
    }

    float scale_h = static_cast<float>(oh) / img_bgr.rows;
    float scale_w = static_cast<float>(ow) / img_bgr.cols;

    auto boxes = postprocess(prob, scale_h, scale_w, img_bgr.size());

    if (ms_out)
        *ms_out = T.toc_ms();

    return boxes;
}

// ---------- utils for IOBinding ----------
std::pair<int, int> DBNet::probe_out_shape_for(int W, int H) {
    std::vector<float> dummy(1LL * 3 * H * W, 0.0f);
    std::vector<int64_t> ishape{1, 3, H, W};

    Ort::Value it = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        dummy.data(), dummy.size(),
        ishape.data(), ishape.size()
    );

    const char *input_names[] = {in_name.c_str()};
    const char *output_names[] = {out_name.c_str()};

    auto out = session.Run(Ort::RunOptions{nullptr}, input_names, &it, 1, output_names, 1);

    auto oshape = out[0].GetTensorTypeAndShapeInfo().GetShape();

    int oh = 0, ow = 0;
    bool ok = false;

    if (oshape.size() == 4 && oshape[0] == 1 && oshape[1] == 1) {
        oh = (int)oshape[2];
        ow = (int)oshape[3];
        ok = true;
    }
    else if (oshape.size() == 4 && oshape[0] == 1 && oshape[3] == 1) {
        oh = (int)oshape[1];
        ow = (int)oshape[2];
        ok = true;
    }
    else if (oshape.size() == 3 && oshape[0] == 1) {
        oh = (int)oshape[1];
        ow = (int)oshape[2];
        ok = true;
    }
    else if (oshape.size() == 2) {
        oh = (int)oshape[0];
        ow = (int)oshape[1];
        ok = true;
    }

    if (!ok)
        throw std::runtime_error("Cannot probe output shape for the given input size.");

    return {ow, oh};
}

void DBNet::prepare_binding(BindingCtx &ctx, int W, int H) {
    if (ctx.bound && ctx.curW == W && ctx.curH == H)
        return;
    
    auto [OW, OH] = probe_out_shape_for(W, H);

    const size_t in_count = 1ull * 3 * H * W;
    const size_t out_count = 1ull * 1 * OH * OW;

    ctx.in_buf.resize(in_count);
    ctx.out_buf.resize(out_count);

    ctx.in_shape = {1, 3, H, W};
    ctx.out_shape = {1, 1, OH, OW};

    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        ctx.mem,
        ctx.in_buf.data(), ctx.in_buf.size(),
        ctx.in_shape.data(), ctx.in_shape.size()
    );

    Ort::Value out_tensor = Ort::Value::CreateTensor<float>(
        ctx.mem,
        ctx.out_buf.data(), ctx.out_buf.size(),
        ctx.out_shape.data(), ctx.out_shape.size()
    );

    ctx.io.ClearBoundInputs();
    ctx.io.ClearBoundOutputs();
    ctx.io.BindInput(in_name.c_str(), in_tensor);
    ctx.io.BindOutput(out_name.c_str(), out_tensor);

    ctx.curW = W;
    ctx.curH = H;
    ctx.curOW = OW;
    ctx.curOH = OH;
    ctx.bound = true;
}

void DBNet::ensure_pool_size(int n) {
    if ((int)pool.size() >= n)
        return;
    for (int i = (int)pool.size(); i < n; ++i)
        pool.emplace_back(std::make_unique<BindingCtx>(session));
}

// ---------- infer_bound with IOBinding ----------
std::vector<Detection> DBNet::infer_bound(const cv::Mat &img_bgr, int ctx_idx, double *ms_out) {
    Timer T;
    T.tic();

    if (ctx_idx < 0 || ctx_idx >= (int)pool.size())
        throw std::runtime_error("Bad binding context index.");
    
    BindingCtx &ctx = *pool[ctx_idx];

    int W = 0, H = 0;
    if (fixed_W > 0 && fixed_H > 0) {
        W = fixed_W;
        H = fixed_H;
    }
    else {
        int h = img_bgr.rows, w = img_bgr.cols;
        float scale = 1.0f;
        int max_side = std::max(h, w);

        if (max_side > limit_side_len)
            scale = static_cast<float>(limit_side_len) / max_side;

        H = std::max(32, (int)std::round(h * scale));
        W = std::max(32, (int)std::round(w * scale));

        H = (H + 31) / 32 * 32;
        W = (W + 31) / 32 * 32;
    }

    prepare_binding(ctx, W, H);

    preprocess_fixed_into(ctx.in_buf.data(), img_bgr, W, H);

    session.Run(Ort::RunOptions{nullptr}, ctx.io);

    cv::Mat prob(ctx.curOH, ctx.curOW, CV_32F, ctx.out_buf.data());
    cv::Mat prob_copy = prob.clone();

    if (apply_sigmoid) {
        cv::Mat ex;
        cv::exp(-prob_copy, ex);
        prob_copy = 1.0f / (1.0f + ex);
    }

    float scale_h = static_cast<float>(ctx.curOH) / img_bgr.rows;
    float scale_w = static_cast<float>(ctx.curOW) / img_bgr.cols;

    auto boxes = postprocess(prob_copy, scale_h, scale_w, img_bgr.size());

    if (ms_out)
        *ms_out = T.toc_ms();

    return boxes;
}
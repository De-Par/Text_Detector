// Production-grade text detector (DBNet/PP-OCR det ONNX)
//
// Key features:
//  • IOBinding with per-thread binding contexts (zero allocs per frame; reuses buffers)
//  • Optional fixed input size (--fixed_hw WxH) for guaranteed stable shapes
//  • Bench mode (--bench, --warmup, --no_draw) with p50/p90/p99
//  • Tiling (RxC) with overlap + polygon NMS (IoU threshold)
//  • Robust ONNX output-shape handling (NCHW/NHWC/2D/3D); optional --apply_sigmoid
//  • OpenMP affinity flags (--omp_places, --omp_bind) + ORT intra-op threads (--threads)
//  • OpenCV threads disabled to avoid oversubscription
//
// Platform: Linux/macOS (Clang/GCC + libomp)

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <cstdlib> // setenv, getenv (POSIX)

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------- Small time helper ----------
struct Timer
{
    std::chrono::steady_clock::time_point t0;
    void tic() { t0 = std::chrono::steady_clock::now(); }
    double toc_ms() const
    {
        using namespace std::chrono;
        return duration<double, std::milli>(steady_clock::now() - t0).count();
    }
};

// ---------- Geometry helpers ----------
static void 
order_quad(cv::Point2f pts[4])
{
    // Ensure consistent vertex order tl, tr, br, bl (clockwise)
    std::vector<cv::Point2f> v(pts, pts + 4);
    std::sort(v.begin(), v.end(), [](const cv::Point2f &a, const cv::Point2f &b)
        { return (a.y < b.y) || (a.y == b.y && a.x < b.x); });

    cv::Point2f tl = v[0].x < v[1].x ? v[0] : v[1];
    cv::Point2f tr = v[0].x < v[1].x ? v[1] : v[0];
    cv::Point2f bl = v[2].x < v[3].x ? v[2] : v[3];
    cv::Point2f br = v[2].x < v[3].x ? v[3] : v[2];

    pts[0] = tl;
    pts[1] = tr;
    pts[2] = br;
    pts[3] = bl;
}

static float 
contour_score(const cv::Mat &prob, const std::vector<cv::Point> &contour)
{
    // Average probability inside the contour (on prob map)
    cv::Rect bbox = cv::boundingRect(contour) & cv::Rect(0, 0, prob.cols, prob.rows);
    if (bbox.empty())
        return 0.f;

    cv::Mat mask = cv::Mat::zeros(bbox.size(), CV_8U);
    std::vector<std::vector<cv::Point>> cnt(1);
    cnt[0].reserve(contour.size());

    for (const auto &p : contour)
        cnt[0].push_back(p - bbox.tl());

    cv::drawContours(mask, cnt, 0, cv::Scalar(255), cv::FILLED);
    cv::Mat roi = prob(bbox);
    cv::Scalar m = cv::mean(roi, mask);

    return static_cast<float>(m[0]);
}

static float 
poly_area(const std::vector<cv::Point2f> &p)
{
    if (p.size() < 3)
        return 0.f;

    double a = 0.0;
    for (size_t i = 0, j = p.size() - 1; i < p.size(); j = i++)
        a += (double)p[j].x * p[i].y - (double)p[i].x * p[j].y;
    
    return static_cast<float>(std::abs(a) * 0.5);
}

static float 
quad_iou(const std::array<cv::Point2f, 4> &A, const std::array<cv::Point2f, 4> &B)
{
    // Convex polygon IoU via OpenCV
    std::vector<cv::Point2f> a(A.begin(), A.end()), b(B.begin(), B.end()), inter;
    float inter_area = (float)cv::intersectConvexConvex(a, b, inter, true);

    if (inter_area <= 0.f)
        return 0.f;

    float ua = poly_area(a) + poly_area(b) - inter_area;
    return ua > 0.f ? inter_area / ua : 0.f;
}

struct Detection {
    std::array<cv::Point2f, 4> pts; // ordered quadrilateral
    float score;                     // mean prob inside contour
};

// ---------- OpenMP affinity helpers ----------
static void 
setenv_if_unset(const char *key, const char *val)
{
    if (!std::getenv(key)) setenv(key, val, 1);
}

static void 
configure_openmp_affinity(const std::string &omp_places_cli, const std::string &omp_bind_cli, int tile_omp_threads)
{
    // Places: from CLI if provided, otherwise default to "cores" (safe baseline)
    if (!omp_places_cli.empty())
        setenv("OMP_PLACES", omp_places_cli.c_str(), 1);
    else
        setenv_if_unset("OMP_PLACES", "cores");

    // Proc bind policy: from CLI or default to "close" (keeps threads near)
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

// ---------- DBNet / PP-OCR det wrapper with IOBinding ----------
//
// We support two execution modes:
//  1) Unbound (simple): create Ort::Value per call (still fast; more allocs)
//  2) Bound (IOBinding): per-thread binding contexts with reusable buffers
//
// For best reuse, you can force a fixed input size via --fixed_hw WxH.
// If not fixed, we cache shape→binding inside each context (first use builds it).

struct DBNet
{
    // Thresholds / params
    float bin_thresh    = 0.3f;
    float box_thresh    = 0.6f;
    int limit_side_len = 960;   // used when not using --fixed_hw
    float unclip_ratio  = 1.5f;
    int min_text_size  = 3;
    bool apply_sigmoid = false; // set if model outputs logits

    // Fixed-size override (optional): if >0, we always resize to (fixed_W, fixed_H)
    int fixed_W = 0, fixed_H = 0;

    // ORT session and options
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dbnet"};
    Ort::Session session{nullptr};
    Ort::SessionOptions so;

    // Common
    Ort::AllocatorWithDefaultOptions alloc;
    std::string in_name, out_name;

    DBNet(const std::string &model_path, int intra_threads = 0, int inter_threads = 1)
    {
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (intra_threads > 0)
            so.SetIntraOpNumThreads(intra_threads);
        if (inter_threads > 0)
            so.SetInterOpNumThreads(inter_threads);
        session = Ort::Session(env, model_path.c_str(), so);

        // Resolve names once
        in_name  = session.GetInputNameAllocated(0, alloc).get();
        out_name = session.GetOutputNameAllocated(0, alloc).get();
    }

    // ---- Preprocess helpers ----

    // A) Dynamic-size preprocess (preserve aspect; scale to limit_side_len; multiple of 32).
    //    Used in simple (unbound) path.
    void 
    preprocess_dynamic(const cv::Mat &img_bgr, cv::Mat &resized, cv::Mat &blob) const
    {
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

        // HWC -> CHW (flat row)
        std::vector<cv::Mat> ch(3);
        cv::split(norm, ch);

        const int hw = resized.rows * resized.cols;
        blob = cv::Mat(1, 3 * hw, CV_32F);

        float *dst = blob.ptr<float>();
        std::memcpy(dst + 0 * hw, ch[0].reshape(1, 1).ptr<float>(), hw * sizeof(float));
        std::memcpy(dst + 1 * hw, ch[1].reshape(1, 1).ptr<float>(), hw * sizeof(float));
        std::memcpy(dst + 2 * hw, ch[2].reshape(1, 1).ptr<float>(), hw * sizeof(float));
    }

    // B) Fixed-size preprocess *into CHW buffer* (no allocations; stretches to WxH).
    //    For speed/IOBinding we do a simple resize (aspect ratio may change).
    //    If you need letterbox, add padding+offset mapping logic (not included here).
    void 
    preprocess_fixed_into(float *dst_chw, const cv::Mat &img_bgr, int W, int H) const
    {
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

    // ---- Postprocess: prob map -> polygons ----
    std::vector<Detection> 
    postprocess(const cv::Mat &prob_map, float scale_h, float scale_w, const cv::Size &orig_size) const
    {
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
        for (auto &c : contours)
        {
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
            for (int i = 0; i < 4; ++i)
            {
                // Map prob-map coords back to original image resolution
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

    // ---------- Unbound (simple) inference ----------
    std::vector<Detection> 
    infer_unbound(const cv::Mat &img_bgr, double *ms_out = nullptr)
    {
        Timer T;
        T.tic();

        cv::Mat resized, blob;
        if (fixed_W > 0 && fixed_H > 0)
        {
            // Forcing fixed size even in unbound mode (still creates a small blob)
            const int hw = fixed_W * fixed_H;
            blob = cv::Mat(1, 3 * hw, CV_32F);
            preprocess_fixed_into(blob.ptr<float>(), img_bgr, fixed_W, fixed_H);
            resized.create(fixed_H, fixed_W, CV_8UC3); // dummy holder to pass dims
            // NOTE: resized is not used further; dims come from fixed_W/H
        }
        else
            preprocess_dynamic(img_bgr, resized, blob);
        
        const int nh = (fixed_H > 0 ? fixed_H : resized.rows);
        const int nw = (fixed_W > 0 ? fixed_W : resized.cols);

        std::vector<int64_t> input_shape = {1, 3, nh, nw};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size());

        const char *input_names[] = {in_name.c_str()};
        const char *output_names[] = {out_name.c_str()};

        auto out = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        if (out.size() != 1)
            throw std::runtime_error("Unexpected number of outputs (expected 1).");

        float *out_data = out[0].GetTensorMutableData<float>();
        auto out_shape = out[0].GetTensorTypeAndShapeInfo().GetShape();

        int oh = 0, ow = 0;
        bool ok = false;
        // Accept NCHW [1,1,H,W]
        if (out_shape.size() == 4 && out_shape[0] == 1 && out_shape[1] == 1)
        {
            oh = (int)out_shape[2];
            ow = (int)out_shape[3];
            ok = true;
        }
        // NHWC [1,H,W,1]
        else if (out_shape.size() == 4 && out_shape[0] == 1 && out_shape[3] == 1)
        {
            oh = (int)out_shape[1];
            ow = (int)out_shape[2];
            ok = true;
        }
        // [1,H,W]
        else if (out_shape.size() == 3 && out_shape[0] == 1)
        {
            oh = (int)out_shape[1];
            ow = (int)out_shape[2];
            ok = true;
        }
        // [H,W]
        else if (out_shape.size() == 2)
        {
            oh = (int)out_shape[0];
            ow = (int)out_shape[1];
            ok = true;
        }
        if (!ok || oh <= 0 || ow <= 0)
            throw std::runtime_error("Unexpected output shape.");

        cv::Mat prob_map(oh, ow, CV_32F, out_data);
        cv::Mat prob = prob_map.clone();
        if (apply_sigmoid)
        {
            cv::Mat ex;
            cv::exp(-prob, ex);
            prob = 1.0f / (1.0f + ex);
        }

        // Map output to original: use direct ratio out/orig (works for both dynamic and fixed resize)
        float scale_h = static_cast<float>(oh) / static_cast<float>(img_bgr.rows);
        float scale_w = static_cast<float>(ow) / static_cast<float>(img_bgr.cols);

        auto boxes = postprocess(prob, scale_h, scale_w, img_bgr.size());

        if (ms_out)
            *ms_out = T.toc_ms();

        return boxes;
    }

    // ---------- IOBinding contexts ----------
    struct BindingCtx
    {
        // One IoBinding per thread; we also cache a single current shape per ctx
        Ort::IoBinding io;
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<float> in_buf;
        std::vector<float> out_buf;
        std::vector<int64_t> in_shape;
        std::vector<int64_t> out_shape;

        int curW = 0, curH = 0, curOW = 0, curOH = 0;
        bool bound = false;

        explicit BindingCtx(Ort::Session &s) : io(s) {}
    };

    // Pool of contexts, indexed by thread id
    std::vector<std::unique_ptr<BindingCtx>> pool;

    void 
    ensure_pool_size(int n)
    {
        if ((int)pool.size() >= n)
            return;
        for (int i = (int)pool.size(); i < n; ++i)
            pool.emplace_back(std::make_unique<BindingCtx>(session));
    }

    // One-time probe to learn output HxW for a given input HxW
    std::pair<int, int> 
    probe_out_shape_for(int W, int H)
    {
        // Build a dummy input tensor (single allocation on stack-like vector)
        std::vector<float> dummy(1LL * 3 * H * W, 0.0f);
        std::vector<int64_t> ishape{1, 3, H, W};
        Ort::Value it = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
            dummy.data(), dummy.size(), ishape.data(), ishape.size());

        const char *input_names[] = {in_name.c_str()};
        const char *output_names[] = {out_name.c_str()};
        auto out = session.Run(Ort::RunOptions{nullptr}, input_names, &it, 1, output_names, 1);
        auto oshape = out[0].GetTensorTypeAndShapeInfo().GetShape();

        int oh = 0, ow = 0;
        bool ok = false;
        if (oshape.size() == 4 && oshape[0] == 1 && oshape[1] == 1)
        {
            oh = (int)oshape[2];
            ow = (int)oshape[3];
            ok = true;
        }
        else if (oshape.size() == 4 && oshape[0] == 1 && oshape[3] == 1)
        {
            oh = (int)oshape[1];
            ow = (int)oshape[2];
            ok = true;
        }
        else if (oshape.size() == 3 && oshape[0] == 1)
        {
            oh = (int)oshape[1];
            ow = (int)oshape[2];
            ok = true;
        }
        else if (oshape.size() == 2)
        {
            oh = (int)oshape[0];
            ow = (int)oshape[1];
            ok = true;
        }

        if (!ok)
            throw std::runtime_error("Cannot probe output shape for the given input size.");

        return {ow, oh};
    }

    // Prepare (or reuse) binding for a specific WxH in this context
    void 
    prepare_binding(BindingCtx &ctx, int W, int H)
    {
        if (ctx.bound && ctx.curW == W && ctx.curH == H)
            return; // already ready

        // Query output size for this input (first time only)
        auto [OW, OH] = probe_out_shape_for(W, H);

        // Allocate buffers once
        const size_t in_count = 1ull * 3 * H * W;
        const size_t out_count = 1ull * 1 * OH * OW;

        ctx.in_buf.resize(in_count);
        ctx.out_buf.resize(out_count);

        ctx.in_shape = {1, 3, H, W};
        ctx.out_shape = {1, 1, OH, OW}; // compatible with NCHW-like; NHWC will also be contiguous

        // Build Ort::Value on top of our buffers
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(ctx.mem, ctx.in_buf.data(), ctx.in_buf.size(),
                                                               ctx.in_shape.data(), ctx.in_shape.size());
        Ort::Value out_tensor = Ort::Value::CreateTensor<float>(ctx.mem, ctx.out_buf.data(), ctx.out_buf.size(),
                                                                ctx.out_shape.data(), ctx.out_shape.size());

        // Re-bind inputs/outputs
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

    // ---------- Bound (IOBinding) inference ----------
    std::vector<Detection> 
    infer_bound(const cv::Mat &img_bgr, int ctx_idx, double *ms_out = nullptr)
    {
        Timer T;
        T.tic();

        if (ctx_idx < 0 || ctx_idx >= (int)pool.size())
            throw std::runtime_error("Bad binding context index.");

        BindingCtx &ctx = *pool[ctx_idx];

        // Choose target size: fixed (if provided) or dynamic (preserve aspect to multiple of 32)
        int W = 0, H = 0;
        if (fixed_W > 0 && fixed_H > 0)
        {
            W = fixed_W;
            H = fixed_H;
        }
        else
        {
            // mimic preprocess_dynamic to maintain shape compatibility with unbound path
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

        // Ensure binding (alloc+bind once per WxH per context)
        prepare_binding(ctx, W, H);

        // Preprocess directly into ctx.in_buf (CHW)
        preprocess_fixed_into(ctx.in_buf.data(), img_bgr, W, H);

        // Run with IOBinding (no new Ort::Value/allocations)
        session.Run(Ort::RunOptions{nullptr}, ctx.io);

        // View result
        cv::Mat prob(ctx.curOH, ctx.curOW, CV_32F, ctx.out_buf.data());
        cv::Mat prob_copy = prob.clone(); // we will modify for thresholding etc.

        if (apply_sigmoid)
        {
            cv::Mat ex;
            cv::exp(-prob_copy, ex);
            prob_copy = 1.0f / (1.0f + ex);
        }

        // Map output to original: direct ratio out/orig
        float scale_h = static_cast<float>(ctx.curOH) / static_cast<float>(img_bgr.rows);
        float scale_w = static_cast<float>(ctx.curOW) / static_cast<float>(img_bgr.cols);

        auto boxes = postprocess(prob_copy, scale_h, scale_w, img_bgr.size());

        if (ms_out)
            *ms_out = T.toc_ms();

        return boxes;
    }
};

// ---------- Drawing + log to stdout (required in spec) ----------
static void 
draw_and_dump(cv::Mat &img, const std::vector<Detection> &dets)
{
    for (const auto &d : dets)
    {
        cv::Point p[4] = {
            cv::Point(cvRound(d.pts[0].x), cvRound(d.pts[0].y)),
            cv::Point(cvRound(d.pts[1].x), cvRound(d.pts[1].y)),
            cv::Point(cvRound(d.pts[2].x), cvRound(d.pts[2].y)),
            cv::Point(cvRound(d.pts[3].x), cvRound(d.pts[3].y)),
        };
        for (int i = 0; i < 4; ++i)
            cv::line(img, p[i], p[(i + 1) % 4], cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        std::cout << p[0].x << "," << p[0].y << " "
                  << p[1].x << "," << p[1].y << " "
                  << p[2].x << "," << p[2].y << " "
                  << p[3].x << "," << p[3].y << std::endl;
    }
}

// ---------- Polygon NMS ----------
static std::vector<Detection> 
nms_poly(const std::vector<Detection> &dets, float iou_thr)
{
    if (dets.empty())
        return {};

    std::vector<int> idx(dets.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return dets[a].score > dets[b].score; });

    std::vector<char> suppressed(dets.size(), 0);
    std::vector<Detection> keep;
    keep.reserve(dets.size());

    for (size_t _i = 0; _i < idx.size(); ++_i)
    {
        int i = idx[_i];
        if (suppressed[i])
            continue;
        keep.push_back(dets[i]);
        for (size_t _j = _i + 1; _j < idx.size(); ++_j)
        {
            int j = idx[_j];
            if (suppressed[j])
                continue;
            float iou = quad_iou(dets[i].pts, dets[j].pts);
            if (iou >= iou_thr)
                suppressed[j] = 1;
        }
    }
    return keep;
}

// ---------- Tiling ----------
struct GridSpec {
    int rows = 1, cols = 1;
};

static bool parse_tiles(const std::string &s, GridSpec &g)
{
    if (s.empty())
        return false;
    size_t pos = s.find_first_of("xX*");
    if (pos == std::string::npos)
        return false;
    int r = 0, c = 0;
    try
    {
        r = std::stoi(s.substr(0, pos));
        c = std::stoi(s.substr(pos + 1));
    }
    catch (...)
    {
        return false;
    }
    if (r <= 0 || c <= 0)
        return false;
    g.rows = r;
    g.cols = c;
    return true;
}

static std::vector<Detection>
infer_tiled_unbound(const cv::Mat &img, DBNet &det, const GridSpec &g, 
        float overlap_frac, double *ms_sum = nullptr, int /*omp_threads*/ = 0)
{
    const int H = img.rows, W = img.cols;
    std::vector<cv::Rect> tiles;
    tiles.reserve(g.rows * g.cols);

    for (int r = 0; r < g.rows; ++r)
    {
        int y0 = (r * H) / g.rows;
        int y1 = ((r + 1) * H) / g.rows;
        for (int c = 0; c < g.cols; ++c)
        {
            int x0 = (c * W) / g.cols;
            int x1 = ((c + 1) * W) / g.cols;

            int tw = x1 - x0, th = y1 - y0;
            int dx = int(std::round(0.5f * overlap_frac * tw));
            int dy = int(std::round(0.5f * overlap_frac * th));

            int ex0 = std::max(0, x0 - dx);
            int ey0 = std::max(0, y0 - dy);
            int ex1 = std::min(W, x1 + dx);
            int ey1 = std::min(H, y1 + dy);

            tiles.emplace_back(ex0, ey0, ex1 - ex0, ey1 - ey0);
        }
    }

    std::vector<Detection> all;
    double sum_ms = 0.0;

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<Detection> local;
        double local_ms = 0.0;
#pragma omp for schedule(dynamic)
        for (int i = 0; i < (int)tiles.size(); ++i)
        {
            const cv::Rect &roi = tiles[i];
            cv::Mat patch = img(roi);
            double ms = 0.0;
            auto dets = det.infer_unbound(patch, &ms);
            for (auto &d : dets)
            {
                for (int k = 0; k < 4; ++k)
                {
                    d.pts[k].x += roi.x;
                    d.pts[k].y += roi.y;
                }
                local.push_back(d);
            }
            local_ms += ms;
        }
#pragma omp critical
        {
            all.insert(all.end(), local.begin(), local.end());
            sum_ms += local_ms;
        }
    }
#else
    for (const auto &roi : tiles)
    {
        cv::Mat patch = img(roi);
        double ms = 0.0;
        auto dets = det.infer_unbound(patch, &ms);
        for (auto &d : dets)
        {
            for (int k = 0; k < 4; ++k)
            {
                d.pts[k].x += roi.x;
                d.pts[k].y += roi.y;
            }
            all.push_back(d);
        }
        sum_ms += ms;
    }
#endif

    if (ms_sum)
        *ms_sum = sum_ms;

    return all;
}

static std::vector<Detection>
infer_tiled_bound(const cv::Mat &img, DBNet &det, const GridSpec &g,
        float overlap_frac, double *ms_sum = nullptr, int omp_threads = 0)
{
    const int H = img.rows, W = img.cols;
    std::vector<cv::Rect> tiles;
    tiles.reserve(g.rows * g.cols);

    for (int r = 0; r < g.rows; ++r)
    {
        int y0 = (r * H) / g.rows;
        int y1 = ((r + 1) * H) / g.rows;
        for (int c = 0; c < g.cols; ++c)
        {
            int x0 = (c * W) / g.cols;
            int x1 = ((c + 1) * W) / g.cols;

            int tw = x1 - x0, th = y1 - y0;
            int dx = int(std::round(0.5f * overlap_frac * tw));
            int dy = int(std::round(0.5f * overlap_frac * th));

            int ex0 = std::max(0, x0 - dx);
            int ey0 = std::max(0, y0 - dy);
            int ex1 = std::min(W, x1 + dx);
            int ey1 = std::min(H, y1 + dy);

            tiles.emplace_back(ex0, ey0, ex1 - ex0, ey1 - ey0);
        }
    }

#ifdef _OPENMP
    int nthreads = (omp_threads > 0 ? omp_threads : omp_get_max_threads());
#else
    int nthreads = 1;
    (void)omp_threads;
#endif
    det.ensure_pool_size(std::max(1, nthreads));

    std::vector<Detection> all;
    double sum_ms = 0.0;

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<Detection> local;
        double local_ms = 0.0;
        int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic)
        for (int i = 0; i < (int)tiles.size(); ++i)
        {
            const cv::Rect &roi = tiles[i];
            cv::Mat patch = img(roi);
            double ms = 0.0;
            auto dets = det.infer_bound(patch, tid, &ms);
            for (auto &d : dets)
            {
                for (int k = 0; k < 4; ++k)
                {
                    d.pts[k].x += roi.x;
                    d.pts[k].y += roi.y;
                }
                local.push_back(d);
            }
            local_ms += ms;
        }
#pragma omp critical
        {
            all.insert(all.end(), local.begin(), local.end());
            sum_ms += local_ms;
        }
    }
#else
    double ms = 0.0;
    auto dets = det.infer_bound(img, 0, &ms);
    all.insert(all.end(), dets.begin(), dets.end());
    sum_ms += ms;
#endif

    if (ms_sum)
        *ms_sum = sum_ms;

    return all;
}

// ---------- CLI / main ----------
static void 
usage(const char *prog)
{
    std::cout << "Usage: " << prog << "\n"
              << "  --model det.onnx --image input.jpg [--out out.png]\n"
              << "  [--bin_thresh 0.3] [--box_thresh 0.6] [--side 960]\n"
              << "  [--threads N] [--unclip 1.5] [--apply_sigmoid 0|1]\n"
              << "  [--tiles RxC] [--tile_overlap 0.10] [--nms_iou 0.3] [--tile_omp N]\n"
              << "  [--omp_places <cores|threads|sockets|{…}>] [--omp_bind <close|spread|master|true|false>]\n"
              << "  [--bind_io 0|1] [--fixed_hw WxH]\n"
              << "  [--bench N] [--warmup K] [--no_draw 0|1]\n";
}

static double 
percentile(std::vector<double> v, double p)
{
    if (v.empty())
        return 0.0;
    if (p < 0)
        p = 0;
    if (p > 1)
        p = 1;
    size_t k = std::min(v.size() - 1, (size_t)std::floor(p * (v.size() - 1)));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

int main(int argc, char **argv)
{
    std::string model_path, image_path, out_path = "out.png";
    float bin_thresh = 0.3f, box_thresh = 0.6f, unclip = 1.5f;
    int side = 960, ort_threads = 0;
    std::string tiles_arg;
    float tile_overlap = 0.10f;
    float nms_iou = 0.30f;
    int tile_omp_threads = 0;
    int apply_sigmoid = 0;

    // OpenMP affinity flags
    std::string omp_places_cli;
    std::string omp_bind_cli;

    // IOBinding / fixed shape
    int bind_io = 0;
    std::string fixed_hw;

    // Bench mode
    int bench_iters = 0, warmup = 20, no_draw = 0;

    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        auto need = [&](const char *key){ return a == key && (i + 1 < argc); };

        if (need("--model"))
            model_path = argv[++i];
        else if (need("--image"))
            image_path = argv[++i];
        else if (need("--out"))
            out_path = argv[++i];
        else if (need("--bin_thresh"))
            bin_thresh = std::stof(argv[++i]);
        else if (need("--box_thresh"))
            box_thresh = std::stof(argv[++i]);
        else if (need("--side"))
            side = std::stoi(argv[++i]);
        else if (need("--threads"))
            ort_threads = std::stoi(argv[++i]);
        else if (need("--unclip"))
            unclip = std::stof(argv[++i]);
        else if (need("--tiles"))
            tiles_arg = argv[++i];
        else if (need("--tile_overlap"))
            tile_overlap = std::stof(argv[++i]);
        else if (need("--nms_iou"))
            nms_iou = std::stof(argv[++i]);
        else if (need("--tile_omp"))
            tile_omp_threads = std::stoi(argv[++i]);
        else if (need("--apply_sigmoid"))
            apply_sigmoid = std::stoi(argv[++i]);
        else if (need("--omp_places"))
            omp_places_cli = argv[++i];
        else if (need("--omp_bind"))
            omp_bind_cli = argv[++i];
        else if (need("--bind_io"))
            bind_io = std::stoi(argv[++i]);
        else if (need("--fixed_hw"))
            fixed_hw = argv[++i];
        else if (need("--bench"))
            bench_iters = std::stoi(argv[++i]);
        else if (need("--warmup"))
            warmup = std::stoi(argv[++i]);
        else if (need("--no_draw"))
            no_draw = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help")
        {
            usage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "Unknown arg: " << a << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    // Parse --fixed_hw WxH if provided
    auto clampf = [](float v, float lo, float hi){ return std::max(lo, std::min(hi, v)); };
    bin_thresh = clampf(bin_thresh, 0.f, 1.f);
    box_thresh = clampf(box_thresh, 0.f, 1.f);
    
    if (tile_overlap < 0.f)
        tile_overlap = 0.f;
    if (tile_overlap > 0.5f)
        tile_overlap = 0.5f;
    if (nms_iou < 0.f)
        nms_iou = 0.f;
    if (nms_iou > 1.f)
        nms_iou = 1.f;
    if (side < 32)
        side = 32;

    int fixedW = 0, fixedH = 0;
    if (!fixed_hw.empty())
    {
        size_t pos = fixed_hw.find_first_of("xX*");
        if (pos == std::string::npos)
        {
            std::cerr << "Bad --fixed_hw format. Use WxH.\n";
            return 1;
        }
        try
        {
            fixedW = std::stoi(fixed_hw.substr(0, pos));
            fixedH = std::stoi(fixed_hw.substr(pos + 1));
        }
        catch (...)
        {
            std::cerr << "Bad --fixed_hw numbers.\n";
            return 1;
        }
        if (fixedW < 32 || fixedH < 32)
        {
            std::cerr << "--fixed_hw too small.\n";
            return 1;
        }
        // Round to 32 multiples (common for det heads)
        fixedW = (fixedW + 31) / 32 * 32;
        fixedH = (fixedH + 31) / 32 * 32;
    }

    // Configure OpenMP affinity BEFORE any OpenMP regions
    configure_openmp_affinity(omp_places_cli, omp_bind_cli, tile_omp_threads);

    // Disable OpenCV internal threading
    cv::setUseOptimized(true);
    cv::setNumThreads(0);

    // --- BENCH fast-path (if requested) ---
    if (bench_iters > 0)
    {
        if (model_path.empty() || image_path.empty())
        {
            usage(argv[0]);
            return 1;
        }
        cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::cerr << "Failed to load image.\n";
            return 2;
        }

        DBNet det(model_path, ort_threads > 0 ? ort_threads : 0);
        det.bin_thresh = bin_thresh;
        det.box_thresh = box_thresh;
        det.limit_side_len = side;
        det.unclip_ratio = unclip;
        det.apply_sigmoid = (apply_sigmoid != 0);
        if (fixedW > 0 && fixedH > 0)
        {
            det.fixed_W = fixedW;
            det.fixed_H = fixedH;
        }

        GridSpec g{1, 1};
        bool use_tiles = parse_tiles(tiles_arg, g);

        // Prepare binding pool if needed
        if (bind_io)
        {
#ifdef _OPENMP
            int nthreads = (tile_omp_threads > 0 ? tile_omp_threads : omp_get_max_threads());
#else
            int nthreads = 1;
#endif
            det.ensure_pool_size(std::max(1, nthreads));
        }

        // Warmup
        for (int i = 0; i < warmup; i++)
        {
            if (use_tiles)
            {
                double ms = 0.0;
                if (bind_io)
                    (void)infer_tiled_bound(img, det, g, tile_overlap, &ms, tile_omp_threads);
                else
                    (void)infer_tiled_unbound(img, det, g, tile_overlap, &ms, tile_omp_threads);
            }
            else
            {
                double ms = 0.0;
                if (bind_io)
                    (void)det.infer_bound(img, 0, &ms);
                else
                    (void)det.infer_unbound(img, &ms);
            }
        }

        // Measure
        std::vector<double> totals, infers;
        totals.reserve(bench_iters);
        infers.reserve(bench_iters);

        for (int i = 0; i < bench_iters; i++)
        {
            Timer T;
            T.tic();
            double infer_ms = 0.0;
            std::vector<Detection> dets;

            if (use_tiles)
            {
                double ms = 0.0;
                if (bind_io)
                    dets = infer_tiled_bound(img, det, g, tile_overlap, &ms, tile_omp_threads);
                else
                    dets = infer_tiled_unbound(img, det, g, tile_overlap, &ms, tile_omp_threads);
                infer_ms = ms; // sum of tile inference times
            }
            else
            {
                if (bind_io)
                    dets = det.infer_bound(img, 0, &infer_ms);
                else
                    dets = det.infer_unbound(img, &infer_ms);
            }

            double total_ms = T.toc_ms();

            if (!no_draw)
            {
                cv::Mat out = img.clone();
                draw_and_dump(out, dets);
                // not saving to disk in bench mode for cleaner timing
            }

            totals.push_back(total_ms);
            infers.push_back(infer_ms);
        }

        auto t50 = percentile(totals, 0.50), t90 = percentile(totals, 0.90), t99 = percentile(totals, 0.99);
        auto i50 = percentile(infers, 0.50), i90 = percentile(infers, 0.90), i99 = percentile(infers, 0.99);
        double avg = 0;
        for (double x : totals)
            avg += x;
        avg /= std::max<size_t>(1, totals.size());
        double fps_p50 = (t50 > 0 ? 1000.0 / t50 : 0.0);

        std::cerr << "[BENCH] iters=" << bench_iters
                  << " total_ms: avg=" << avg << " p50=" << t50 << " p90=" << t90 << " p99=" << t99
                  << " | infer_ms: p50=" << i50 << " p90=" << i90 << " p99=" << i99
                  << " | fps@p50=" << fps_p50
                  << " | bind_io=" << bind_io
                  << " | tiles=" << (use_tiles ? tiles_arg : "1x1")
                  << " | ORT_threads=" << (ort_threads > 0 ? ort_threads : 1)
                  << std::endl;
        return 0;
    }

    // --- Normal single-shot run ---
    if (model_path.empty() || image_path.empty())
    {
        usage(argv[0]);
        return 1;
    }
    try
    {
        cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
        if (img.empty())
            throw std::runtime_error("Failed to load image: " + image_path);

        DBNet det(model_path, ort_threads > 0 ? ort_threads : 0);
        det.bin_thresh = bin_thresh;
        det.box_thresh = box_thresh;
        det.limit_side_len = side;
        det.unclip_ratio = unclip;
        det.apply_sigmoid = (apply_sigmoid != 0);
        if (fixedW > 0 && fixedH > 0)
        {
            det.fixed_W = fixedW;
            det.fixed_H = fixedH;
        }

        GridSpec g{1, 1};
        bool use_tiles = parse_tiles(tiles_arg, g);

        double infer_ms = 0.0;
        std::vector<Detection> dets;
        Timer T;
        T.tic();

        if (use_tiles)
        {
            if (bind_io)
            {
#ifdef _OPENMP
                int nthreads = (tile_omp_threads > 0 ? tile_omp_threads : omp_get_max_threads());
#else
                int nthreads = 1;
#endif
                det.ensure_pool_size(std::max(1, nthreads));
                dets = infer_tiled_bound(img, det, g, tile_overlap, &infer_ms, tile_omp_threads);
            }
            else
            {
                dets = infer_tiled_unbound(img, det, g, tile_overlap, &infer_ms, tile_omp_threads);
            }
            dets = nms_poly(dets, nms_iou);
        }
        else
        {
            if (bind_io)
            {
                det.ensure_pool_size(1);
                dets = det.infer_bound(img, 0, &infer_ms);
            }
            else
            {
                dets = det.infer_unbound(img, &infer_ms);
            }
        }

        double total_ms = T.toc_ms();

        cv::Mat out = img.clone();
        draw_and_dump(out, dets);
        if (!out_path.empty())
            cv::imwrite(out_path, out);

        std::cerr << "[OK] dets=" << dets.size()
                  << " | infer_ms=" << infer_ms
                  << " | total_ms=" << total_ms
#ifdef _OPENMP
                  << " | OMP:on"
#else
                  << " | OMP:off"
#endif
                  << " | ORT_threads=" << (ort_threads > 0 ? ort_threads : 1)
                  << " | bind_io=" << bind_io
                  << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
#include <algorithm>
#include <numeric>
#include "nms.h"
#include "geometry.h"

std::vector<Detection> nms_poly(const std::vector<Detection> &dets, float iou_thr)
{
    if (dets.empty())
        return {};

    std::vector<int> idx(dets.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return dets[a].score > dets[b].score;
    });

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
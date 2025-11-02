#pragma once
#include <vector>
#include "geometry.h"

std::vector<Detection> nms_poly(const std::vector<Detection> &dets,
                                float iou_thr);

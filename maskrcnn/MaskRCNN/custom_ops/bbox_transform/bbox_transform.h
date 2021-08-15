// Copyright 2004-present Facebook. All Rights Reserved.

#include <torch/csrc/api/include/torch/types.h>
#include "generate_proposals_op_util_boxes.h"

namespace caffe2 {
namespace fb {

std::tuple<torch::Tensor, torch::Tensor> BBoxTransformCPUKernel(
    const torch::Tensor& roi_in,
    const torch::Tensor& delta_in,
    const torch::Tensor& iminfo_in,
    std::vector<double> weights_,
    bool apply_scale_,
    bool rotated_,
    bool angle_bound_on_,
    int64_t angle_bound_lo_,
    int64_t angle_bound_hi_,
    double clip_angle_thresh_,
    bool legacy_plus_one_,
    c10::optional<std::vector<torch::Tensor>> /* unused */ = {});

} // namespace fb
} // namespace caffe2

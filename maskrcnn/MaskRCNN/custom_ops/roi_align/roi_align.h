// Copyright 2004-present Facebook. All Rights Reserved.

#include "roi_align-cpu.h"
#include <torch/csrc/api/include/torch/types.h>

namespace caffe2 {
namespace fb {

torch::Tensor RoIAlignCPUKernel(
    const torch::Tensor& features,
    const torch::Tensor& rois,
    std::string order,
    double spatial_scale,
    int64_t aligned_height,
    int64_t aligned_width,
    int64_t sampling_ratio,
    bool aligned,
    c10::optional<std::vector<torch::Tensor>>);

} // namespace fb
} // namespace caffe2

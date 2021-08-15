// Copyright 2004-present Facebook. All Rights Reserved.

#include "roi_align.h"

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
    c10::optional<std::vector<torch::Tensor>>) {
  TORCH_CHECK(order == "NCHW" || order == "NHWC");
  return torch::fb::RoIAlignForwardCPUKernel(
      features,
      rois,
      aligned_height,
      aligned_width,
      spatial_scale,
      sampling_ratio,
      aligned);
}

} // namespace fb
} // namespace caffe2

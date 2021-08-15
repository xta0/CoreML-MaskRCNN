#ifndef ROI_ALIGN_CPU_H
#define ROI_ALIGN_CPU_H

#include <torch/script.h>

namespace torch {
namespace fb {

torch::Tensor RoIAlignForwardCPUKernel(
    const torch::Tensor& features,
    const torch::Tensor& rois,
    int64_t aligned_height,
    int64_t aligned_width,
    double spatial_scale,
    int64_t sampling_ratio,
    bool aligned);

torch::Tensor RoIAlignBackwardCPUKernel(
    const torch::Tensor& features,
    const torch::Tensor& rois,
    const torch::Tensor& grad_output,
    double spatial_scale,
    int64_t pooled_h,
    int64_t pooled_w,
    int64_t sampling_ratio,
    bool aligned);

} // namespace fb
} // namespace torch

#endif

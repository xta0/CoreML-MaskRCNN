#include "roi_align_impl-cpu.h"
#include "roi_align-cpu.h"

namespace torch {
namespace fb {

torch::Tensor RoIAlignForwardCPUKernel(
    const torch::Tensor& features,
    const torch::Tensor& rois,
    int64_t aligned_height,
    int64_t aligned_width,
    double spatial_scale,
    int64_t sampling_ratio,
    bool aligned) {
  TORCH_CHECK(features.dim(), 4);
  TORCH_CHECK(rois.dim(), 2);
  const int64_t roi_cols = rois.size(1);
  TORCH_CHECK(roi_cols == 4 || roi_cols == 5);
  const int64_t N = rois.size(0);
  const int64_t C = features.size(1);
  const int64_t H = features.size(2);
  const int64_t W = features.size(3);
  const int64_t pooled_h = aligned_height;
  const int64_t pooled_w = aligned_width;

  if (features.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    auto output = torch::empty(
        {N, C, pooled_h, pooled_w},
        features.options(),
        at::MemoryFormat::ChannelsLast);
    if (N == 0) {
      return output;
    }
    torch::fb::ROIAlignForwardCpuImplWithNHWC(
        N,
        C,
        H,
        W,
        roi_cols,
        aligned_height,
        aligned_width,
        spatial_scale,
        sampling_ratio,
        aligned,
        features.data_ptr<float>(),
        rois.contiguous().data_ptr<float>(),
        output.data_ptr<float>());
    return output;
  } else {
    auto features_contig = features.contiguous();
    auto output = torch::empty({N, C, pooled_h, pooled_w}, features.options());
    if (N == 0) {
      return output;
    }
    torch::fb::ROIAlignForwardCpuImplWithNCHW(
        N,
        C,
        H,
        W,
        roi_cols,
        aligned_height,
        aligned_width,
        spatial_scale,
        sampling_ratio,
        aligned,
        features_contig.data_ptr<float>(),
        rois.contiguous().data_ptr<float>(),
        output.data_ptr<float>());
    return output;
  }
}

torch::Tensor RoIAlignBackwardCPUKernel(
    const torch::Tensor& features,
    const torch::Tensor& rois,
    const torch::Tensor& grad_output,
    double spatial_scale,
    int64_t pooled_h,
    int64_t pooled_w,
    int64_t sampling_ratio,
    bool aligned) {
  TORCH_CHECK(rois.dim() == 2);
  const int64_t N = rois.size(0);
  const int64_t roi_cols = rois.size(1);
  const int64_t C = features.size(1);
  const int64_t H = features.size(2);
  const int64_t W = features.size(3);
  auto output = torch::zeros(features.sizes(), features.options());
  const float* dY = grad_output.data_ptr<float>();
  const float* R = rois.data_ptr<float>();
  float* dX = output.data_ptr<float>();
  if (grad_output.numel() >
      0) { // Handle possibly empty gradient if there were no rois
    torch::fb::ROIAlignBackwardCpuImplWithNCHW(
        grad_output.numel(),
        dY,
        N,
        spatial_scale,
        C,
        H,
        W,
        pooled_h,
        pooled_w,
        sampling_ratio,
        dX,
        R,
        roi_cols,
        aligned);
  }
  return output;
}

} // namespace fb
} // namespace torch

#if !defined(C10_MOBILE)
TORCH_LIBRARY_FRAGMENT(fb, m) {
  m.def(
      "roi_align_forward("
      " Tensor features,"
      " Tensor rois,"
      " int aligned_height,"
      " int aligned_width,"
      " float spatial_scale,"
      " int sample_ratio,"
      " bool aligned"
      ") -> Tensor");
  m.impl(
      "roi_align_forward",
      torch::dispatch(
          c10::DispatchKey::CPU,
          TORCH_FN(torch::fb::RoIAlignForwardCPUKernel)));
  m.def(
      "roi_align_backward("
      " Tensor features,"
      " Tensor rois,"
      " Tensor grad_output,"
      " float spatial_scale,"
      " int pooled_h,"
      " int pooled_w,"
      " int sample_ratio,"
      " bool aligned"
      ") -> Tensor");
  m.impl(
      "roi_align_backward",
      torch::dispatch(
          c10::DispatchKey::CPU,
          TORCH_FN(torch::fb::RoIAlignBackwardCPUKernel)));
}
#endif

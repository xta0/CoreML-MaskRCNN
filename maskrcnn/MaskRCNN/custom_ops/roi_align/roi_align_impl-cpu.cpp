// Copyright 2004-present Facebook. All Rights Reserved.

#include "roi_align_impl-cpu.h"
#include "eigen_utils.h"

namespace torch {
namespace fb {

namespace {

template <typename T>
std::vector<BilinearInterpolationParam<float>> MakeBilinearInterpolationParams(
    int64_t H,
    int64_t W,
    int64_t pooled_h,
    int64_t pooled_w,
    T bin_size_h,
    T bin_size_w,
    int64_t bin_grid_h,
    int64_t bin_grid_w,
    T roi_start_h,
    T roi_start_w) {
  std::vector<BilinearInterpolationParam<float>> params(
      pooled_h * pooled_w * bin_grid_h * bin_grid_w);
  const T ch = bin_size_h / static_cast<float>(bin_grid_h);
  const T cw = bin_size_w / static_cast<float>(bin_grid_w);
  int64_t cnt = 0;
  for (int64_t ph = 0; ph < pooled_h; ++ph) {
    for (int64_t pw = 0; pw < pooled_w; ++pw) {
      for (int64_t iy = 0; iy < bin_grid_h; ++iy) {
        const T yy = roi_start_h + static_cast<float>(ph) * bin_size_h +
            (static_cast<float>(iy) + T(0.5)) * ch;
        if (yy < T(-1) || yy > static_cast<float>(H)) {
          std::memset(params.data() + cnt, 0, bin_grid_w * sizeof(params[0]));
          cnt += bin_grid_w;
          continue;
        }
        for (int64_t ix = 0; ix < bin_grid_w; ++ix) {
          const T xx = roi_start_w + pw * bin_size_w +
              (static_cast<float>(ix) + T(0.5f)) * cw;
          BilinearInterpolationParam<float>& param = params[cnt++];
          if (xx < T(-1) || xx > static_cast<float>(W)) {
            std::memset(&param, 0, sizeof(param));
            continue;
          }
          const T y = std::min(std::max(yy, T(0)), static_cast<float>(H - 1));
          const T x = std::min(std::max(xx, T(0)), static_cast<float>(W - 1));
          const int64_t yl = static_cast<int64_t>(std::floor(y));
          const int64_t xl = static_cast<int64_t>(std::floor(x));
          const int64_t yh = std::min(yl + 1, H - 1);
          const int64_t xh = std::min(xl + 1, W - 1);
          const T py = y - static_cast<float>(yl);
          const T px = x - static_cast<float>(xl);
          const T qy = T(1) - py;
          const T qx = T(1) - px;
          param.p1 = yl * W + xl;
          param.p2 = yl * W + xh;
          param.p3 = yh * W + xl;
          param.p4 = yh * W + xh;
          param.w1 = qy * qx;
          param.w2 = qy * px;
          param.w3 = py * qx;
          param.w4 = py * px;
        }
      }
    }
  }
  return params;
}

template <typename T>
void bilinear_interpolate_gradient(
    const int height,
    const int width,
    T y,
    T x,
    T& w1,
    T& w2,
    T& w3,
    T& w4,
    int& x_low,
    int& x_high,
    int& y_low,
    int& y_high,
    const int /*index*/ /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <class T>
inline void add(const T& val, T* address) {
  *address += val;
}

} // namespace

void ROIAlignForwardCpuImplWithNCHW(
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t roi_cols,
    int64_t pooled_h,
    int64_t pooled_w,
    double spatial_scale,
    int64_t sampling_ratio,
    bool aligned,
    const float* X,
    const float* R,
    float* Y) {
  assert(roi_cols == 4 || roi_cols == 5);
  const float roi_offset = aligned ? 0.5f : 0.0f;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int64_t n = 0; n < N; ++n) {
    const int64_t roi_batch_idx = roi_cols == 4 ? 0 : R[n * roi_cols];
    const float* X_ptr = X + roi_batch_idx * C * H * W;
    const float* R_ptr = R + n * roi_cols + (roi_cols == 5);
    float* Y_ptr = Y + n * C * pooled_h * pooled_w;

    // Do not using rounding; this implementation detail is critical
    const float roi_w1 = R_ptr[0] * spatial_scale - roi_offset;
    const float roi_h1 = R_ptr[1] * spatial_scale - roi_offset;
    const float roi_w2 = R_ptr[2] * spatial_scale - roi_offset;
    const float roi_h2 = R_ptr[3] * spatial_scale - roi_offset;
    float roi_w = roi_w2 - roi_w1;
    float roi_h = roi_h2 - roi_h1;
    if (aligned) {
      assert(roi_w >= 0.0f && roi_h >= 0.0f);

    } else { // backward compatibility
      // Force malformed ROIs to be 1x1
      roi_w = std::max(roi_w, 1.0f);
      roi_h = std::max(roi_h, 1.0f);
    }
    const float bin_size_h = roi_h / static_cast<float>(pooled_h);
    const float bin_size_w = roi_w / static_cast<float>(pooled_w);

    // We use roi_bin_grid to sample the grid and mimic integral
    const int64_t bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : static_cast<int64_t>(ceil(roi_h / static_cast<float>(pooled_h)));
    const int64_t bin_grid_w = (sampling_ratio > 0)
        ? sampling_ratio
        : static_cast<int64_t>(ceil(roi_w / static_cast<float>(pooled_w)));

    const std::vector<BilinearInterpolationParam<float>> params =
        MakeBilinearInterpolationParams(
            H,
            W,
            pooled_h,
            pooled_w,
            bin_size_h,
            bin_size_w,
            bin_grid_h,
            bin_grid_w,
            roi_h1,
            roi_w1);

    const float scale = 1.0f / static_cast<float>(bin_grid_h * bin_grid_w);
    for (int64_t c = 0; c < C; ++c) {
      int64_t cnt = 0;
      for (int64_t ph = 0; ph < pooled_h; ++ph) {
        for (int64_t pw = 0; pw < pooled_w; ++pw) {
          float sum = 0.0f;
          for (int64_t iy = 0; iy < bin_grid_h; ++iy) {
            for (int64_t ix = 0; ix < bin_grid_w; ++ix) {
              const BilinearInterpolationParam<float>& param = params[cnt++];
              sum += param.w1 * X_ptr[param.p1] + param.w2 * X_ptr[param.p2] +
                  param.w3 * X_ptr[param.p3] + param.w4 * X_ptr[param.p4];
            }
          }
          Y_ptr[ph * pooled_w + pw] = sum * scale;
        }
      }
      X_ptr += H * W;
      Y_ptr += pooled_h * pooled_w;
    }
  }
}

void ROIAlignForwardCpuImplWithNHWC(
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t roi_cols,
    int64_t pooled_h,
    int64_t pooled_w,
    double spatial_scale,
    int64_t sampling_ratio,
    bool aligned,
    const float* X,
    const float* R,
    float* Y) {
  assert(roi_cols == 4 || roi_cols == 5);
  const float roi_offset = aligned ? 0.5f : 0.0f;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int64_t n = 0; n < N; ++n) {
    const int64_t roi_batch_idx = roi_cols == 4 ? 0 : R[n * roi_cols];
    const float* X_ptr = X + roi_batch_idx * C * H * W;
    const float* R_ptr = R + n * roi_cols + ((roi_cols == 5) ? 1 : 0);
    float* Y_ptr = Y + n * C * pooled_h * pooled_w;

    // Do not using rounding; this implementation detail is critical
    const float roi_w1 = R_ptr[0] * spatial_scale - roi_offset;
    const float roi_h1 = R_ptr[1] * spatial_scale - roi_offset;
    const float roi_w2 = R_ptr[2] * spatial_scale - roi_offset;
    const float roi_h2 = R_ptr[3] * spatial_scale - roi_offset;
    float roi_w = roi_w2 - roi_w1;
    float roi_h = roi_h2 - roi_h1;
    if (aligned) {
      assert(roi_w >= 0.0f && roi_h >= 0.0f);

    } else { // backward compatibility
      // Force malformed ROIs to be 1x1
      roi_w = std::max(roi_w, 1.0f);
      roi_h = std::max(roi_h, 1.0f);
    }
    const float bin_size_h = roi_h / static_cast<float>(pooled_h);
    const float bin_size_w = roi_w / static_cast<float>(pooled_w);

    // We use roi_bin_grid to sample the grid and mimic integral
    const int64_t bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : static_cast<int64_t>(ceil(roi_h / static_cast<float>(pooled_h)));
    const int64_t bin_grid_w = (sampling_ratio > 0)
        ? sampling_ratio
        : static_cast<int64_t>(ceil(roi_w / static_cast<float>(pooled_w)));

    const std::vector<BilinearInterpolationParam<float>> params =
        MakeBilinearInterpolationParams(
            H,
            W,
            pooled_h,
            pooled_w,
            bin_size_h,
            bin_size_w,
            bin_grid_h,
            bin_grid_w,
            roi_h1,
            roi_w1);

    const float scale = 1.0f / static_cast<float>(bin_grid_h * bin_grid_w);
    int64_t cnt = 0;
    for (int64_t ph = 0; ph < pooled_h; ++ph) {
      for (int64_t pw = 0; pw < pooled_w; ++pw) {
        caffe2::EigenVectorArrayMap<float> Y_arr(
            Y_ptr + (ph * pooled_w + pw) * C, C);
        Y_arr.setZero();
        for (int64_t iy = 0; iy < bin_grid_h; ++iy) {
          for (int64_t ix = 0; ix < bin_grid_w; ++ix) {
            const BilinearInterpolationParam<float>& param = params[cnt++];
            caffe2::ConstEigenVectorArrayMap<float> x1_arr(
                X_ptr + param.p1 * C, C);
            caffe2::ConstEigenVectorArrayMap<float> x2_arr(
                X_ptr + param.p2 * C, C);
            caffe2::ConstEigenVectorArrayMap<float> x3_arr(
                X_ptr + param.p3 * C, C);
            caffe2::ConstEigenVectorArrayMap<float> x4_arr(
                X_ptr + param.p4 * C, C);
            Y_arr += param.w1 * x1_arr + param.w2 * x2_arr + param.w3 * x3_arr +
                param.w4 * x4_arr;
          }
        }
        Y_arr *= scale;
      }
    }
  }
}

void ROIAlignBackwardCpuImplWithNCHW(
    const int nthreads,
    const float* top_diff,
    const int /*num_rois*/,
    const float& spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    float* bottom_diff,
    const float* bottom_rois,
    int rois_cols,
    bool continuous_coordinate) {
  assert(rois_cols == 4 || rois_cols == 5);
  for (int index = 0; index < nthreads; index++) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const float* offset_bottom_rois = bottom_rois + n * rois_cols;
    int roi_batch_ind = 0;
    if (rois_cols == 5) {
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++;
    }

    // Do not using rounding; this implementation detail is critical
    float roi_offset = continuous_coordinate ? float(0.5) : 0;
    float roi_start_w = offset_bottom_rois[0] * spatial_scale - roi_offset;
    float roi_start_h = offset_bottom_rois[1] * spatial_scale - roi_offset;
    float roi_end_w = offset_bottom_rois[2] * spatial_scale - roi_offset;
    float roi_end_h = offset_bottom_rois[3] * spatial_scale - roi_offset;

    float roi_width = roi_end_w - roi_start_w;
    float roi_height = roi_end_h - roi_start_h;
    if (continuous_coordinate) {
      assert(roi_width >= 0 && roi_height >= 0);
    } else { // backward compatibility
      // Force malformed ROIs to be 1x1
      roi_width = std::max(roi_width, (float)1.);
      roi_height = std::max(roi_height, (float)1.);
    }
    float bin_size_h =
        static_cast<float>(roi_height) / static_cast<float>(pooled_height);
    float bin_size_w =
        static_cast<float>(roi_width) / static_cast<float>(pooled_width);

    float* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const float* offset_top_diff = top_diff + top_offset;
    const float top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const float count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const float y = roi_start_h + ph * bin_size_h +
          static_cast<float>(iy + .5f) * bin_size_h /
              static_cast<float>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const float x = roi_start_w + pw * bin_size_w +
            static_cast<float>(ix + .5f) * bin_size_w /
                static_cast<float>(roi_bin_grid_w);

        float w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(
            height,
            width,
            y,
            x,
            w1,
            w2,
            w3,
            w4,
            x_low,
            x_high,
            y_low,
            y_high,
            index);

        float g1 = top_diff_this_bin * w1 / count;
        float g2 = top_diff_this_bin * w2 / count;
        float g3 = top_diff_this_bin * w3 / count;
        float g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          // atomic add is not needed for now since it is single threaded
          add(static_cast<float>(g1),
              offset_bottom_diff + y_low * width + x_low);
          add(static_cast<float>(g2),
              offset_bottom_diff + y_low * width + x_high);
          add(static_cast<float>(g3),
              offset_bottom_diff + y_high * width + x_low);
          add(static_cast<float>(g4),
              offset_bottom_diff + y_high * width + x_high);
        } // if
      } // ix
    } // iy
  } // for
}

} // namespace fb
} // namespace torch

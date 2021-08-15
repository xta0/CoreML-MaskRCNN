//
//  generate_proposals.hpp
//  MaskRCNN
//
//  Created by Tao Xu on 8/13/21.
//

#ifndef generate_proposals_hpp
#define generate_proposals_hpp

#include <LibTorch/LibTorch.h>

// Copyright 2004-present Facebook. All Rights Reserved.

#include <torch/csrc/api/include/torch/types.h>
#include "generate_proposals_op_util_boxes.h"
#include "generate_proposals_op_util_nms.h"
#include "eigen_utils.h"

namespace caffe2 {
namespace fb {

std::tuple<torch::Tensor, torch::Tensor> GenerateProposalsCPUKernel(
    const torch::Tensor& scores,
    const torch::Tensor& bbox_deltas,
    const torch::Tensor& im_info_tensor,
    const torch::Tensor& anchors_tensor,
    double spatial_scale_,
    int64_t rpn_pre_nms_topN_,
    int64_t post_nms_topN_,
    double nms_thresh_,
    double rpn_min_size_,
    bool angle_bound_on_,
    int64_t angle_bound_lo_,
    int64_t angle_bound_hi_,
    double clip_angle_thresh_,
    bool legacy_plus_one_,
    c10::optional<std::vector<torch::Tensor>> /* unused */ = {});
} // namespace fb
} // namespace caffe2


#endif /* generate_proposals_hpp */

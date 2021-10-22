// Copyright 2004-present Facebook. All Rights Reserved.

#import "GenerateProposals.h"
#import "multiArray.h"

#include "generate_proposals.h"
#include <caffe2/core/timer.h>

@implementation GenerateProposals {
    double spatial_scale_;
    int64_t rpn_pre_nms_topN_;
    int64_t post_nms_topN_;
    double nms_thresh_;
    double rpn_min_size_;
    bool angle_bound_on_;
    int64_t angle_bound_lo_;
    int64_t angle_bound_hi_;
    double clip_angle_thresh_;
    bool legacy_plus_one_;
}

- (BOOL)evaluateOnCPUWithInputs:(nonnull NSArray<MLMultiArray *> *)inputs outputs:(nonnull NSArray<MLMultiArray *> *)outputs error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    
    at::Tensor scores         = tensorFromMultiArray(inputs[0], 4);
    at::Tensor bbox_deltas    = tensorFromMultiArray(inputs[1], 4);
    at::Tensor im_info_tensor = tensorFromMultiArray(inputs[2], 2);
    at::Tensor anchors_tensor = tensorFromMultiArray(inputs[3], 2);
    
    caffe2::Timer t;
    auto result = caffe2::fb::GenerateProposalsCPUKernel(
          scores,
          bbox_deltas,
          im_info_tensor,
          anchors_tensor,
          spatial_scale_,
          rpn_pre_nms_topN_,
          post_nms_topN_,
          nms_thresh_,
          rpn_min_size_,
          angle_bound_on_,
          angle_bound_lo_,
          angle_bound_hi_,
          clip_angle_thresh_,
          legacy_plus_one_,
          {});
    std::cout<<"[Generate Proposals] took: "<<t.MilliSeconds()<<" ms"<<std::endl;
    auto out_rois = std::get<0>(result);
    auto out_rois_prob = std::get<1>(result);
    memcpy(outputs[0].dataPointer, out_rois.data_ptr<float>(), out_rois.numel() * sizeof(float));
    memcpy(outputs[1].dataPointer, out_rois_prob.data_ptr<float>(), out_rois_prob.numel() * sizeof(float));
    return YES;
}

- (nullable instancetype)initWithParameterDictionary:(nonnull NSDictionary<NSString *,id> *)parameters error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    self = [super init];
    if(self){
        spatial_scale_      = [parameters[@"spatial_scale"] doubleValue];
        rpn_pre_nms_topN_   = [parameters[@"rpn_pre_nms_topN"] intValue];
        post_nms_topN_      = [parameters[@"rpn_post_nms_topN"] intValue];
        nms_thresh_         = [parameters[@"nms_thresh"] doubleValue];
        rpn_min_size_       = [parameters[@"rpn_min_size"] doubleValue];
        angle_bound_on_     = [parameters[@"angle_bound_on"] boolValue];
        angle_bound_lo_     = [parameters[@"angle_bound_lo"] intValue];
        angle_bound_hi_     = [parameters[@"angle_bound_hi"] intValue];
        clip_angle_thresh_  = [parameters[@"clip_angle_thresh"] doubleValue];
        legacy_plus_one_    = [parameters[@"legacy_plus_one"] boolValue];
    }
    return self;
}

- (nullable NSArray<NSArray<NSNumber *> *> *)outputShapesForInputShapes:(nonnull NSArray<NSArray<NSNumber *> *> *)inputShapes error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    TORCH_CHECK(post_nms_topN_ <= 10)
    return @[@[@(post_nms_topN_), @(5)],@[@(post_nms_topN_)]];
}

- (BOOL)setWeightData:(nonnull NSArray<NSData *> *)weights error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    return true;
}

@end

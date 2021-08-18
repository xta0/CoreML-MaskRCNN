// Copyright 2004-present Facebook. All Rights Reserved.

#include "BBoxTransform.h"
#include "bbox_transform.h"

#include <vector>
#include <torch/script.h>

#import "multiArray.h"

@implementation BBoxTransform
{
    std::vector<double> weights_;
    bool apply_scale_;
    bool rotated_;
    bool angle_bound_on_;
    int64_t angle_bound_lo_;
    int64_t angle_bound_hi_;
    double clip_angle_thresh_;
    bool legacy_plus_one_;
}

- (at::Tensor)tensorFromMLMultiArray1:(MLMultiArray* )array API_AVAILABLE(ios(11.0)){
    std::vector<int64_t> size;
    for(NSNumber* n in array.shape){
        size.push_back(n.intValue);
    }
    std::vector<int64_t> sz(size.end()-1, size.end());
    auto tensor = at::empty(sz, c10::kFloat);
    memcpy(tensor.data_ptr<float>(), array.dataPointer, tensor.numel() * sizeof(float));
    return tensor;
}

- (at::Tensor)tensorFromMLMultiArray2:(MLMultiArray* )array API_AVAILABLE(ios(11.0)){
    std::vector<int64_t> size;
    for(NSNumber* n in array.shape){
        size.push_back(n.intValue);
    }
    std::vector<int64_t> sz(size.end()-2, size.end());
    auto tensor = at::empty(sz, c10::kFloat);
    memcpy(tensor.data_ptr<float>(), array.dataPointer, tensor.numel() * sizeof(float));
    return tensor;
}


- (BOOL)evaluateOnCPUWithInputs:(nonnull NSArray<MLMultiArray *> *)inputs outputs:(nonnull NSArray<MLMultiArray *> *)outputs error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    
    at::Tensor roi_in       = tensorFromMultiArray(inputs[0], 2);
    at::Tensor delta_in     = tensorFromMultiArray(inputs[1], 2);
    at::Tensor iminfo_in    = tensorFromMultiArray(inputs[2], 2);
    at::Tensor weights      = tensorFromMultiArray(inputs[3], 1);
    TORCH_CHECK(weights.numel() == 4)
    weights_ = {};
    for(int i=0; i<4; ++i){
        weights_.push_back(weights.data_ptr<float>()[i]);
    }
    auto results = caffe2::fb::BBoxTransformCPUKernel(roi_in,
                                                      delta_in,
                                                      iminfo_in,
                                                      weights_,
                                                      apply_scale_,
                                                      rotated_,
                                                      angle_bound_on_,
                                                      angle_bound_lo_,
                                                      angle_bound_hi_,
                                                      clip_angle_thresh_,
                                                      legacy_plus_one_);
    auto box_out          = std::get<0>(results);
    auto roi_batch_splits = std::get<1>(results);
    MLMultiArray* output0 = outputs[0];
    MLMultiArray* output1 = outputs[1];
    TORCH_CHECK(box_out.numel() == output0.count);
    TORCH_CHECK(roi_batch_splits.numel() == output1.count);
    memcpy(output0.dataPointer, box_out.data_ptr<float>(), box_out.numel() * sizeof(float));
    memcpy(output1.dataPointer, roi_batch_splits.data_ptr<float>(), roi_batch_splits.numel() * sizeof(float));
    return YES;
}

- (nullable instancetype)initWithParameterDictionary:(nonnull NSDictionary<NSString *,id> *)parameters error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    self = [super init];
    if(self){
        apply_scale_        = [parameters[@"apply_scale"] boolValue];
        rotated_            = [parameters[@"rotated"] boolValue];
        angle_bound_on_     = [parameters[@"angle_bound_on"] boolValue];
        angle_bound_lo_     = [parameters[@"angle_bound_lo"] intValue];
        angle_bound_hi_     = [parameters[@"angle_bound_hi"] intValue];
        clip_angle_thresh_  = [parameters[@"clip_angle_thresh"] doubleValue];
        legacy_plus_one_    = [parameters[@"legacy_plus_one"] boolValue];
    }
    return self;
}

- (nullable NSArray<NSArray<NSNumber *> *> *)outputShapesForInputShapes:(nonnull NSArray<NSArray<NSNumber *> *> *)inputShapes error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    NSArray* deltaIn  = inputShapes[1];
    NSArray* iminfoIN = inputShapes[2];
    return @[deltaIn, @[iminfoIN[0]]];
}

- (BOOL)setWeightData:(nonnull NSArray<NSData *> *)weights error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    return true;
}


@end

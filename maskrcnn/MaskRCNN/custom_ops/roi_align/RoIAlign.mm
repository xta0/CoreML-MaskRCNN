// Copyright 2004-present Facebook. All Rights Reserved.

#include "RoIAlign.h"
#include "roi_align.h"
#include <torch/script.h>
#import "multiArray.h"

@implementation RoIAlign {
    double spatial_scale_;
    int64_t aligned_height_;
    int64_t aligned_width_;
    int64_t sampling_ratio_;
    bool aligned_;
}

- (BOOL)evaluateOnCPUWithInputs:(nonnull NSArray<MLMultiArray *> *)inputs outputs:(nonnull NSArray<MLMultiArray *> *)outputs error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    
    at::Tensor features = tensorFromMultiArray(inputs[0], 4);
    at::Tensor rois     = tensorFromMultiArray(inputs[1], 2);
    NSDate* date = [NSDate date];
    auto result = caffe2::fb::RoIAlignCPUKernel(features,
                                                rois,
                                                "NCHW",
                                                spatial_scale_,
                                                aligned_height_,
                                                aligned_width_,
                                                sampling_ratio_,
                                                aligned_,
                                                {});
    NSLog(@"[RoIAlign] took: %.2fms", [date timeIntervalSinceNow] * -1000);
    memcpy(outputs[0].dataPointer, result.data_ptr<float>(), result.numel() * sizeof(float));
    return YES;
}

- (nullable instancetype)initWithParameterDictionary:(nonnull NSDictionary<NSString *,id> *)parameters error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    self = [super init];
    if(self){
        spatial_scale_  = [parameters[@"spatial_scale"] doubleValue];
        aligned_height_ = [parameters[@"aligned_height"] intValue];
        aligned_width_  = [parameters[@"aligned_width"] intValue];
        sampling_ratio_ = [parameters[@"sampling_ratio"] intValue];
        aligned_        = [parameters[@"aligned"] boolValue];
    }
    return self;
}

- (nullable NSArray<NSArray<NSNumber *> *> *)outputShapesForInputShapes:(nonnull NSArray<NSArray<NSNumber *> *> *)inputShapes error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    NSArray* featureShape = inputShapes[0];
    NSArray* roiShape = inputShapes[1];
    NSArray* outputSize = @[roiShape[0], featureShape[1], @(aligned_height_),@(aligned_width_)];
    return @[outputSize];
}

- (BOOL)setWeightData:(nonnull NSArray<NSData *> *)weights error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    return true;
}

@end

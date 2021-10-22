// Copyright 2004-present Facebook. All Rights Reserved.

#import "BoxWithNMSLimit.h"
#import "multiArray.h"

#include "box_with_nms_limit.h"
#include <caffe2/core/timer.h>

@implementation BoxWithNMSLimit {
    double score_thres_;
    double nms_thres_;
    int64_t detections_per_im_;
    bool soft_nms_enabled_;
    std::string soft_nms_method_str_;
    double soft_nms_sigma_;
    double soft_nms_min_score_thres_;
    bool rotated_;
    bool cls_agnostic_bbox_reg_;
    bool input_boxes_include_bg_cls_;
    bool output_classes_include_bg_cls_;
    bool legacy_plus_one_;
}

- (BOOL)evaluateOnCPUWithInputs:(nonnull NSArray<MLMultiArray *> *)inputs outputs:(nonnull NSArray<MLMultiArray *> *)outputs error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    
    at::Tensor tscores       = tensorFromMultiArray(inputs[0], 2);
    at::Tensor tboxes        = tensorFromMultiArray(inputs[1], 2);
    at::Tensor tbatch_splits = tensorFromMultiArray(inputs[2], 1);
    
    caffe2::Timer t;
    auto results = caffe2::fb::BoxWithNMSLimitCPUKernel(tscores,
                                                        tboxes,
                                                        tbatch_splits,
                                                        score_thres_,
                                                        nms_thres_,
                                                        detections_per_im_,
                                                        soft_nms_enabled_,
                                                        soft_nms_method_str_,
                                                        soft_nms_sigma_,
                                                        soft_nms_min_score_thres_,
                                                        rotated_,
                                                        cls_agnostic_bbox_reg_,
                                                        input_boxes_include_bg_cls_,
                                                        output_classes_include_bg_cls_,
                                                        legacy_plus_one_);
    std::cout<<"[BoxWithNMSLimit] took: "<<t.MilliSeconds()<<" ms"<<std::endl;
    auto out_scores = std::get<0>(results);
    auto out_boxes = std::get<1>(results);
    auto out_classes = std::get<2>(results);
    auto batch_split_out = std::get<3>(results);
    auto out_keeps = std::get<4>(results);
    auto out_keeps_size = std::get<5>(results);
    
    memset(outputs[0].dataPointer, 0, outputs[0].count  * sizeof(float));
    memcpy(outputs[0].dataPointer, out_scores.data_ptr<float>(), out_scores.numel()  * sizeof(float));
    
    memset(outputs[1].dataPointer, 0, outputs[1].count  * sizeof(float));
    memcpy(outputs[1].dataPointer, out_boxes.data_ptr<float>(), out_boxes.numel()  * sizeof(float));
    
    memset(outputs[2].dataPointer, 0, outputs[2].count  * sizeof(float));
    memcpy(outputs[2].dataPointer, out_classes.data_ptr<float>(), out_classes.numel()  * sizeof(float));

    return YES;
}

- (nullable instancetype)initWithParameterDictionary:(nonnull NSDictionary<NSString *,id> *)parameters error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    self = [super init];
    if(self){
        score_thres_        = [parameters[@"score_thres"] doubleValue];
        nms_thres_          = [parameters[@"nms_thres"] doubleValue];
        detections_per_im_  = [parameters[@"detections_per_im"] intValue];
        soft_nms_enabled_   = [parameters[@"soft_nms_enabled"] boolValue];
        soft_nms_method_str_ = ((NSString* )parameters[@"soft_nms_method_str"]).UTF8String;
        soft_nms_sigma_     = [parameters[@"soft_nms_sigma"] doubleValue];
        soft_nms_min_score_thres_     = [parameters[@"soft_nms_min_score_thres"] doubleValue];
        rotated_ = [parameters[@"rotated"] boolValue];
        cls_agnostic_bbox_reg_ = [parameters[@"cls_agnostic_bbox_reg"] boolValue];
        input_boxes_include_bg_cls_ = [parameters[@"input_boxes_include_bg_cls"] boolValue];
        output_classes_include_bg_cls_ = [parameters[@"output_classes_include_bg_cls"] boolValue];
        legacy_plus_one_ = [parameters[@"legacy_plus_one"] boolValue];
    }
    return self;
}

- (nullable NSArray<NSArray<NSNumber *> *> *)outputShapesForInputShapes:(nonnull NSArray<NSArray<NSNumber *> *> *)inputShapes
                                                                  error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    NSArray* out_scores = @[@(10)];
    NSArray* out_boxes = @[@(10), @(4)];
    NSArray* out_classes = @[@(10)];
    NSArray* batch_split = @[@(1)];
    NSArray* out_keeps = @[@(10)];
    NSArray* out_keeps_size = @[@(1), @(2)];
    return @[out_scores, out_boxes, out_classes, batch_split, out_keeps, out_keeps_size];
}

- (BOOL)setWeightData:(nonnull NSArray<NSData *> *)weights error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    return true;
}

@end

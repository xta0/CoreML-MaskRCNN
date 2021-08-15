#include "RoIAlign.h"
#include "roi_align.h"
#include <torch/script.h>

@implementation RoIAlign {
    double spatial_scale_;
    int64_t aligned_height_;
    int64_t aligned_width_;
    int64_t sampling_ratio_;
    bool aligned_;
}

- (at::Tensor)tensorFromMLMultiArray:(MLMultiArray* )array API_AVAILABLE(ios(11.0)){
    std::vector<int64_t> size;
    for(NSNumber* n in array.shape){
        size.push_back(n.intValue);
    }
    size.erase(size.begin());
    auto tensor = at::empty(size, c10::kFloat);
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

- (BOOL)evaluateOnCPUWithInputs:(nonnull NSArray<MLMultiArray *> *)inputs outputs:(nonnull NSArray<MLMultiArray *> *)outputs error:(NSError *__autoreleasing  _Nullable * _Nullable)error  API_AVAILABLE(ios(11.0)){
    auto features = [self tensorFromMLMultiArray:inputs[0]];
    auto rois = [self tensorFromMLMultiArray2:inputs[1]];
    auto result = caffe2::fb::RoIAlignCPUKernel(features,
                                                rois,
                                                "NCHW",
                                                spatial_scale_,
                                                aligned_height_,
                                                aligned_width_,
                                                sampling_ratio_,
                                                aligned_,
                                                {});
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

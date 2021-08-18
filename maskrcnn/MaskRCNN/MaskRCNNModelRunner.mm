//
//  MaskRCNNModelRunner.m
//  MaskRCNN
//
//  Created by Tao Xu on 8/13/21.
//

#import "MaskRCNNModelRunner.h"
#import "utils.h"

#import <CoreML/CoreML.h>

#import <opencv2/core/core.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/imgproc/imgproc.hpp>
#import <opencv2/imgproc/types_c.h>

#include <LibTorch/LibTorch.h>

#include <vector>
#include <string>

const float BBOX_SCORE_THRESHOLD = 0.5;

@interface PTMCoreMLFeatureProvider : NSObject<MLFeatureProvider>
- (instancetype)initWithFeatureSpecs:(const std::vector<FeatureSpecs>&)specs
                       CoreMLVersion:(int)ver;
@end

@implementation PTMCoreMLFeatureProvider {
    std::vector<FeatureSpecs> _specs;
}

@synthesize featureNames = _featureNames;

- (instancetype)initWithFeatureSpecs:(const std::vector<FeatureSpecs>&)specs {
    self = [super init];
    if (self) {
        _specs = specs;
        NSMutableArray* names = [NSMutableArray new];
        for (auto& spec : _specs) {
            NSString* name = [NSString stringWithCString:spec.name.c_str()
                                                encoding:NSUTF8StringEncoding];
            [names addObject:name];
        }
        _featureNames = [[NSSet alloc] initWithArray:names];
    }
    return self;
}

- (nullable MLFeatureValue*)featureValueForName:(NSString*)featureName {
    if([featureName isEqualToString:@"input_0"]){
        auto& spec = _specs[0];
        NSMutableArray* shape = [NSMutableArray new];
        for (auto& dim : spec.tensor.sizes().vec()) {
            [shape addObject:@(dim)];
        }
        NSMutableArray* strides = [NSMutableArray new];
        for (auto& step : spec.tensor.strides().vec()) {
            [strides addObject:@(step)];
        }
        NSError* error = nil;
        MLMultiArray* mlArray =
        [[MLMultiArray alloc] initWithDataPointer:spec.tensor.data_ptr<float>()
                                            shape:shape
                                         dataType:MLMultiArrayDataTypeFloat32
                                          strides:strides
                                      deallocator:(^(void* bytes){
        })error:&error];
        return [MLFeatureValue featureValueWithMultiArray:mlArray];
    } else {
        auto& spec = _specs[1];
        NSMutableArray* shape = [NSMutableArray new];
        for (auto& dim : spec.tensor.sizes().vec()) {
            [shape addObject:@(dim)];
        }
        NSMutableArray* strides = [NSMutableArray new];
        for (auto& step : spec.tensor.strides().vec()) {
            [strides addObject:@(step)];
        }
        NSError* error = nil;
        MLMultiArray* mlArray =
        [[MLMultiArray alloc] initWithDataPointer:spec.tensor.data_ptr<float>()
                                            shape:shape
                                         dataType:MLMultiArrayDataTypeFloat32
                                          strides:strides
                                      deallocator:(^(void* bytes){
        })error:&error];
        return [MLFeatureValue featureValueWithMultiArray:mlArray];
    }
}

@end

@implementation MaskRCNNModelRunner {
    MLModel* _mlModel;
}

- (UIImage *)processImage:(CVImageBufferRef)imageBuffer
{
    CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
    const int planeIndex = 0;
    void *baseAddress =
    CVPixelBufferGetBaseAddressOfPlane(imageBuffer, planeIndex);
    size_t orig_width = CVPixelBufferGetWidth(imageBuffer);
    size_t orig_height = CVPixelBufferGetHeight(imageBuffer);
    size_t stride = CVPixelBufferGetBytesPerRowOfPlane(imageBuffer, planeIndex);
    // resize the image
    CGSize size = calculateSize(orig_height, orig_width);
    cv::Mat matBgra = cv::Mat(orig_height, orig_width, CV_8UC4, (void *)baseAddress, stride);
    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
    cv::Mat matResizedBgra;
    cv::resize(matBgra, matResizedBgra, cv::Size(size.width, size.height));
    cv::Mat matResizedRgb;
    cv::cvtColor(matResizedBgra, matResizedRgb, CV_BGRA2RGB);
    std::vector<float> bgrBuffer;
    convertBGRAToBGRPlane(
                          matResizedBgra.data,
                          bgrBuffer,
                          (int)size.width,
                          (int)size.height
                          );
    at::Tensor input0 = torch::from_blob(bgrBuffer.data(), {1, 3, (int)size.height, (int)size.width}, at::kFloat);
    std::vector<float> imageSize{(float)size.height, (float)size.width, 1.0};
    at::Tensor input1 = torch::from_blob(imageSize.data(), {1, 3});
    // input is a tuple
    auto results = [self _run:{
        {
            .name = "input_0",
            .tensor = input0,
        },
        {
            .name = "input_1",
            .tensor = input1
        }
    }];
    // bbox, bbox scores, class, mask heat maps, and keypoints
    
    // collect bbox scores
    std::vector<float> scores;
    auto bboxScores = results[0];
    float *ptr = bboxScores.data_ptr<float>();
    scores.reserve(bboxScores.numel());
    for (int i = 0; i < bboxScores.numel(); ++i) {
        scores.push_back(ptr[i]);
    }
    
    // collect bounding boxes
    std::vector<CGRect> bboxes;
    auto bboxTensor = results[1]; // [N, 4]
    int64_t bboxNum = bboxTensor.size(0);
    
    // collect masks
    NSMutableArray<UIImage *> *masks = [NSMutableArray new];
    auto maskTensor = results[3]; // [n, 1, 12, 12]
    for (int i = 0; i < bboxNum; ++i) {
        if (scores[i] < BBOX_SCORE_THRESHOLD) {
            continue;
        }
        auto bbox = bboxTensor[i];
        ptr = bbox.data_ptr<float>();
        float x = ptr[0];
        float y = ptr[1];
        float w = ptr[2] - x;
        float h = ptr[3] - y;
        bboxes.push_back(CGRectMake(x, y, w, h));
        
        auto mask = maskTensor[i][0];
        std::vector<UInt8> maskGrayScale(mask.numel());
        for (int j = 0; j < mask.numel(); ++j) {
            int pixel = (int)(mask.data_ptr<float>()[j] * 255.0);
            maskGrayScale[j] = UInt8(pixel);
        }
        UIColor *color = [UIColor colorWithHue:0.55 + i * 0.20 saturation:1.0
                                    brightness:1.0 alpha:1.0];
        CGFloat red, green, blue, alpha;
        [color getRed:&red green:&green blue:&blue alpha:&alpha];
        int targetHeight = h;
        int targetWidth = w;
        cv::Mat b(targetHeight, targetWidth, CV_8UC1);
        cv::Mat g(targetHeight, targetWidth, CV_8UC1);
        cv::Mat r(targetHeight, targetWidth, CV_8UC1);
        b = cv::Scalar(blue * 255.0);
        g = cv::Scalar(green * 255.0);
        r = cv::Scalar(red * 255.0);
        cv::Mat a(28, 28, CV_8UC1, maskGrayScale.data());
        cv::resize(a, a, cv::Size(w, h));
        cv::threshold(a, a, 128, 128, 0);
        std::vector<cv::Mat> ch = {b, g, r, a};
        cv::Mat mat;
        cv::merge(ch, mat);
        UIImage *maskImage = MatToUIImage(mat);
        [masks addObject:maskImage];
    }
    UIImage *resizedRGBImage = MatToUIImage(matResizedRgb);
    return drawMasks(resizedRGBImage, bboxes, masks);
}

- (void)loadModel {
    NSError* error;
    MLModelConfiguration* config = [MLModelConfiguration alloc];
    config.computeUnits = MLComputeUnitsCPUAndGPU;
    config.allowLowPrecisionAccumulationOnGPU = YES;
    NSString* path = [[NSBundle mainBundle] pathForResource:@"maskrcnn_oss_coreml.mlmodelc" ofType:nil];
    _mlModel = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path]
                                 configuration:config
                                         error:&error];
}

- (UIImage* )run {
    UIImage *image = [UIImage imageWithContentsOfFile:[[NSBundle mainBundle] pathForResource:@"maskrcnn" ofType:@"jpg"]];
    CVImageBufferRef imageBuffer = PixelBufferCreateFromCGImage(image.CGImage);
    UIImage *output = [self processImage:imageBuffer];
    CVPixelBufferRelease(imageBuffer);
    return output;
}


- (std::vector<at::Tensor>)_run:(const std::vector<FeatureSpecs>&)inputs {
    NSError* error = nil;
    PTMCoreMLFeatureProvider* inputFeature =
    [[PTMCoreMLFeatureProvider alloc] initWithFeatureSpecs:inputs];
    MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
    NSDate* date = [NSDate date];
    id<MLFeatureProvider> outputFeatures =
    [_mlModel predictionFromFeatures:inputFeature
                             options:options
                               error:&error];
    NSLog(@"took: %.2fms", [date timeIntervalSinceNow] * -1000);
    std::vector<NSString*> outputNames {
        @"boxwithnmslimit_0:0",
        @"boxwithnmslimit_0:1",
        @"boxwithnmslimit_0:2",
        @"680"
    };
    std::vector<std::vector<int64_t>> outputSizes {
        {10},
        {10, 4},
        {10},
        {10, 80, 28, 28}
    };
    std::vector<at::Tensor> outputs;
    for(int i=0; i< outputNames.size(); ++i) {
        MLFeatureValue* val = [outputFeatures featureValueForName:outputNames[i]];
        auto tensor = at::empty(outputSizes[i]);
        int64_t count = val.multiArrayValue.count;
        float* ptr = (float*)val.multiArrayValue.dataPointer;
        memcpy(tensor.data_ptr<float>(), ptr, count * sizeof(float));
        outputs.push_back(tensor);
    }
    return outputs;
}

@end

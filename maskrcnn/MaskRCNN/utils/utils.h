//
//  utils.h
//  MaskRCNN
//
//  Created by Tao Xu on 8/14/21.
//

#ifndef utils_h
#define utils_h

#import <CoreVideo/CVPixelBuffer.h>
#import <UIKit/UIKit.h>
#include <vector>
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static inline UIColor *randomColor()
{
    CGFloat r = arc4random_uniform(256) / 255.0;
    CGFloat g = arc4random_uniform(256) / 255.0;
    CGFloat b = arc4random_uniform(256) / 255.0;
    return [UIColor colorWithRed:r green:g blue:b alpha:1.0];
}

static inline CGSize calculateSize(int h, int w)
{
    // resize shortest edge
    int min_size = 224;
    int max_size = 448;
    int size = min_size;
    float scale = size * 1.0 / MIN(h, w);
    float newh = 0, neww = 0;
    if (h < w) {
        newh = size;
        neww = scale * w;
    } else {
        neww = size;
        newh = scale * h;
    }
    if (MAX(newh, neww) > max_size) {
        scale = max_size * 1.0 / MAX(newh, neww);
        newh = newh * scale;
        neww = neww * scale;
    }
    return CGSizeMake((int)(neww + 0.5), (int)(newh + 0.5));
}

static UIImage *resize(UIImage *image, CGSize newSize)
{
    UIGraphicsBeginImageContextWithOptions(newSize, NO, 0.0);
    [image drawInRect:CGRectMake(0, 0, newSize.width, newSize.height)];
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    NSString *imagePath = [NSHomeDirectory() stringByAppendingPathComponent:@"Documents/maskrcnn_resize.png"];
    NSLog(@"%@", imagePath);
    [UIImagePNGRepresentation(newImage) writeToFile:imagePath atomically:YES];
    return newImage;
}

static std::vector<float> BGRPlane(UIImage* image){
    CGImageRef inputCGImage = image.CGImage;
    NSUInteger width = CGImageGetWidth(inputCGImage);
    NSUInteger height = CGImageGetHeight(inputCGImage);
    NSUInteger bytesPerPixel = 4;
    NSUInteger bytesPerRow = bytesPerPixel * width;
    NSUInteger bitsPerComponent = 8;
    std::vector<uint8_t> rawPixels(width *height * bytesPerPixel);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context =
    CGBitmapContextCreate(
                          rawPixels.data(),
                          width,
                          height,
                          bitsPerComponent,
                          bytesPerRow,
                          colorSpace,
                          kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
                          );
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), inputCGImage);
    CGColorSpaceRelease(colorSpace);
    CGContextRelease(context);
    std::vector<float> buffer(height * width * 3);
    for (NSUInteger i = 0; i < height * width; ++i) {
        buffer[i] = rawPixels[i * 4 + 2];
        buffer[width * height + i] = rawPixels[i * 4 + 1];
        buffer[width * height * 2 + i] = rawPixels[i * 4];
    }
    return buffer;
}

static UIImage *drawMasks(UIImage *image,
                          const std::vector<CGRect> &bboxes,
                          NSArray<UIImage *> *masks) {
    CGSize size = CGSizeMake(image.size.width, image.size.height);
    UIGraphicsBeginImageContext(size);
    CGContextRef context = UIGraphicsGetCurrentContext();
    [image drawInRect:CGRectMake(0, 0, size.width, size.height)];
    CGContextSetLineWidth(context, 2.0f);
    for (int i = 0; i < bboxes.size(); ++i) {
        CGRect box = bboxes[i];
        if (box.size.width > 0 && box.size.height > 0) {
            CGContextSetStrokeColorWithColor(context, [randomColor() CGColor]);
            CGContextStrokeRect(context, CGRectMake(box.origin.x, box.origin.y, box.size.width, box.size.height));
            // draw mask
            [masks[i] drawInRect:CGRectMake(box.origin.x, box.origin.y, box.size.width, box.size.height)];
        }
    }
    UIImage *img = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return img;
}

inline void convertBGRAToBGRPlaneWithDimensionsOffsetAndScale(const uint8_t* bgraPtr,
                                                              std::vector<float>& bgrPlane,
                                                              const float scaleDownHeight,
                                                              const float scaleDownWidth,
                                                              const float RGBNormFactor,
                                                              const float* BGRChannelOffset,
                                                              const float RGBScaleFactor) {
    bgrPlane.resize(3 * scaleDownWidth * scaleDownHeight);
    
    // Convert to planar BGR in the form BBGGRR
    int b_idx = 0, // Start of B plane.
    g_idx = scaleDownWidth * scaleDownHeight, // Start of G plane
    r_idx = 2 * scaleDownWidth * scaleDownHeight; // Start of R plane
    
    int i = 0;
    
    // Clang LLVM with Ofast enabled optimizes this better on IOS
#if (defined(__ARM_NEON__) || defined(__ARM_NEON)) && \
!(defined(__clang__) && defined(__APPLE__))
    const float norm_scale = RGBNormFactor * RGBScaleFactor;
    bool normalized = (norm_scale == 1.0f);
    const float32x4_t b_offset =
    vdupq_n_f32(BGRChannelOffset[0] * RGBScaleFactor);
    const float32x4_t g_offset =
    vdupq_n_f32(BGRChannelOffset[1] * RGBScaleFactor);
    const float32x4_t r_offset =
    vdupq_n_f32(BGRChannelOffset[2] * RGBScaleFactor);
    
    for (; i < 4 * scaleDownWidth * scaleDownHeight - 64;
         i += 64, r_idx += 16, g_idx += 16, b_idx += 16) {
        uint8x16x4_t dt = vld4q_u8(bgraPtr + i);
        
        float32x4x4_t b = convertu8tof32(dt.val[0]);
        float32x4x4_t g = convertu8tof32(dt.val[1]);
        float32x4x4_t r = convertu8tof32(dt.val[2]);
        
        if (normalized) {
            vst1q_f32(bgrPlane.data() + b_idx, vaddq_f32(b.val[0], b_offset));
            vst1q_f32(bgrPlane.data() + b_idx + 4, vaddq_f32(b.val[1], b_offset));
            vst1q_f32(bgrPlane.data() + b_idx + 8, vaddq_f32(b.val[2], b_offset));
            vst1q_f32(bgrPlane.data() + b_idx + 12, vaddq_f32(b.val[3], b_offset));
            
            vst1q_f32(bgrPlane.data() + g_idx, vaddq_f32(g.val[0], g_offset));
            vst1q_f32(bgrPlane.data() + g_idx + 4, vaddq_f32(g.val[1], g_offset));
            vst1q_f32(bgrPlane.data() + g_idx + 8, vaddq_f32(g.val[2], g_offset));
            vst1q_f32(bgrPlane.data() + g_idx + 12, vaddq_f32(g.val[3], g_offset));
            
            vst1q_f32(bgrPlane.data() + r_idx, vaddq_f32(r.val[0], r_offset));
            vst1q_f32(bgrPlane.data() + r_idx + 4, vaddq_f32(r.val[1], r_offset));
            vst1q_f32(bgrPlane.data() + r_idx + 8, vaddq_f32(r.val[2], r_offset));
            vst1q_f32(bgrPlane.data() + r_idx + 12, vaddq_f32(r.val[3], r_offset));
        } else {
            vst1q_f32(
                      bgrPlane.data() + b_idx,
                      vaddq_f32(vmulq_n_f32(b.val[0], norm_scale), b_offset));
            vst1q_f32(
                      bgrPlane.data() + b_idx + 4,
                      vaddq_f32(vmulq_n_f32(b.val[1], norm_scale), b_offset));
            vst1q_f32(
                      bgrPlane.data() + b_idx + 8,
                      vaddq_f32(vmulq_n_f32(b.val[2], norm_scale), b_offset));
            vst1q_f32(
                      bgrPlane.data() + b_idx + 12,
                      vaddq_f32(vmulq_n_f32(b.val[3], norm_scale), b_offset));
            
            vst1q_f32(
                      bgrPlane.data() + g_idx,
                      vaddq_f32(vmulq_n_f32(g.val[0], norm_scale), g_offset));
            vst1q_f32(
                      bgrPlane.data() + g_idx + 4,
                      vaddq_f32(vmulq_n_f32(g.val[1], norm_scale), g_offset));
            vst1q_f32(
                      bgrPlane.data() + g_idx + 8,
                      vaddq_f32(vmulq_n_f32(g.val[2], norm_scale), g_offset));
            vst1q_f32(
                      bgrPlane.data() + g_idx + 12,
                      vaddq_f32(vmulq_n_f32(g.val[3], norm_scale), g_offset));
            
            vst1q_f32(
                      bgrPlane.data() + r_idx,
                      vaddq_f32(vmulq_n_f32(r.val[0], norm_scale), r_offset));
            vst1q_f32(
                      bgrPlane.data() + r_idx + 4,
                      vaddq_f32(vmulq_n_f32(r.val[1], norm_scale), r_offset));
            vst1q_f32(
                      bgrPlane.data() + r_idx + 8,
                      vaddq_f32(vmulq_n_f32(r.val[2], norm_scale), r_offset));
            vst1q_f32(
                      bgrPlane.data() + r_idx + 12,
                      vaddq_f32(vmulq_n_f32(r.val[3], norm_scale), r_offset));
        }
    }
#endif
    
    for (; i < 4 * scaleDownWidth * scaleDownHeight;
         i += 4, r_idx++, g_idx++, b_idx++) {
        bgrPlane.data()[b_idx] =
        (bgraPtr[i + 0] * RGBNormFactor + BGRChannelOffset[0]) * RGBScaleFactor;
        bgrPlane.data()[g_idx] =
        (bgraPtr[i + 1] * RGBNormFactor + BGRChannelOffset[1]) * RGBScaleFactor;
        bgrPlane.data()[r_idx] =
        (bgraPtr[i + 2] * RGBNormFactor + BGRChannelOffset[2]) * RGBScaleFactor;
    }
}

// A flavor of the conversion that does not do any BGR offsetting. Models using this should have in-model normalization
static inline void convertBGRAToBGRPlaneForARTracking(const uint8_t* bgraPtr, std::vector<float>& bgrPlane, int width, int height) {
    float scaleDownWidth = 0.0f, scaleDownHeight = 0.0f;
    float BGRChannelOffset[3] = {0.0f, 0.0f, 0.0f};
    scaleDownHeight = height;
    scaleDownWidth = width;
    convertBGRAToBGRPlaneWithDimensionsOffsetAndScale(
                                                      bgraPtr, bgrPlane, scaleDownHeight, scaleDownWidth, 1.0, BGRChannelOffset, 1.0);
}

static inline NSDictionary *DefaultPixelBufferAttributes()
{
    const CFStringRef openGLCompatibilityKey = kCVPixelBufferOpenGLCompatibilityKey;
    NSDictionary *attributes = @{(__bridge id)kCVPixelBufferIOSurfacePropertiesKey : @{},
                                 (__bridge id)openGLCompatibilityKey : @YES,
                                 (__bridge id)kCVPixelBufferMetalCompatibilityKey : @YES, };
    
    return attributes;
}
static inline CVPixelBufferRef FBPixelBufferCreateFromResizedCGImage(CGImageRef image, CGSize outputSize)
{
    CVPixelBufferRef pxbuffer = NULL;
    CVReturn status = CVPixelBufferCreate(
                                          kCFAllocatorDefault,
                                          outputSize.width,
                                          outputSize.height,
                                          kCVPixelFormatType_32BGRA,
                                          (__bridge CFDictionaryRef)DefaultPixelBufferAttributes(),
                                          &pxbuffer
                                          );
    if (status != kCVReturnSuccess) {
        if (pxbuffer != NULL) {
            CVPixelBufferRelease(pxbuffer);
        }
        return NULL;
    }
    
    CVPixelBufferLockBaseAddress(pxbuffer, 0);
    void *pxdata = CVPixelBufferGetBaseAddress(pxbuffer);
    
    CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(
                                                 pxdata,
                                                 outputSize.width,
                                                 outputSize.height,
                                                 8,
                                                 CVPixelBufferGetBytesPerRow(pxbuffer),
                                                 rgbColorSpace,
                                                 (CGBitmapInfo)kCGBitmapByteOrder32Little
                                                 | kCGImageAlphaPremultipliedFirst
                                                 );
    
    CGContextDrawImage(context, CGRectMake(0, 0, outputSize.width, outputSize.height), image);
    CGColorSpaceRelease(rgbColorSpace);
    CGContextRelease(context);
    
    CVPixelBufferUnlockBaseAddress(pxbuffer, 0);
    
    return pxbuffer;
}

static inline CVPixelBufferRef FBPixelBufferCreateFromCGImage(CGImageRef image)
{
    const CGSize frameSize = CGSizeMake(CGImageGetWidth(image), CGImageGetHeight(image));
    return FBPixelBufferCreateFromResizedCGImage(image, frameSize);
}

#endif /* utils_h */

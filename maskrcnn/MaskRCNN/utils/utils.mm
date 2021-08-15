//
//  utils.m
//  MaskRCNN
//
//  Created by Tao Xu on 8/14/21.
//

#import "utils.h"
#import <Foundation/Foundation.h>
#import <opencv2/core/core.hpp>
#import <opencv2/imgproc/imgproc.hpp>

//NSDictionary *DefaultPixelBufferAttributes()
//{
//  const CFStringRef openGLCompatibilityKey = kCVPixelBufferOpenGLCompatibilityKey;
//  NSDictionary *attributes = @{(__bridge id)kCVPixelBufferIOSurfacePropertiesKey : @{},
//                               (__bridge id)openGLCompatibilityKey : @YES,
//                               (__bridge id)kCVPixelBufferMetalCompatibilityKey : @YES, };
//
//  return attributes;
//}
//
//CVPixelBufferRef FBPixelBufferCreateFromResizedCGImage(CGImageRef image, CGSize outputSize)
//{
//  CVPixelBufferRef pxbuffer = NULL;
//  CVReturn status = CVPixelBufferCreate(
//    kCFAllocatorDefault,
//    outputSize.width,
//    outputSize.height,
//    kCVPixelFormatType_32BGRA,
//    (__bridge CFDictionaryRef)DefaultPixelBufferAttributes(),
//    &pxbuffer
//  );
//  if (status != kCVReturnSuccess) {
//    if (pxbuffer != NULL) {
//      CVPixelBufferRelease(pxbuffer);
//    }
//    return NULL;
//  }
//
//  CVPixelBufferLockBaseAddress(pxbuffer, 0);
//  void *pxdata = CVPixelBufferGetBaseAddress(pxbuffer);
//
//  CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
//  CGContextRef context = CGBitmapContextCreate(
//    pxdata,
//    outputSize.width,
//    outputSize.height,
//    8,
//    CVPixelBufferGetBytesPerRow(pxbuffer),
//    rgbColorSpace,
//    (CGBitmapInfo)kCGBitmapByteOrder32Little
//    | kCGImageAlphaPremultipliedFirst
//  );
//
//  CGContextDrawImage(context, CGRectMake(0, 0, outputSize.width, outputSize.height), image);
//  CGColorSpaceRelease(rgbColorSpace);
//  CGContextRelease(context);
//
//  CVPixelBufferUnlockBaseAddress(pxbuffer, 0);
//
//  return pxbuffer;
//}
//
//static CVPixelBufferRef FBPixelBufferCreateFromCGImage(CGImageRef image) {
//    const CGSize frameSize = CGSizeMake(CGImageGetWidth(image), CGImageGetHeight(image));
//    return FBPixelBufferCreateFromResizedCGImage(image, frameSize);
//}

//static UIImage *drawMasks(UIImage *image,
//                          const std::vector<CGRect> &bboxes,
//                          NSArray<UIImage *> *masks) {
//    
//}

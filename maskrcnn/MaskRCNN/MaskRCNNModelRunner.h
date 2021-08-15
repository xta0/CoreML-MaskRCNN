//
//  MaskRCNNModelRunner.h
//  MaskRCNN
//
//  Created by Tao Xu on 8/13/21.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#include <LibTorch/LibTorch.h>
#include <string>
#include <vector>

NS_ASSUME_NONNULL_BEGIN

struct FeatureSpecs {
  std::string name;
  at::Tensor tensor;
};

@interface MaskRCNNModelRunner : NSObject

- (void)loadModel;
- (UIImage* )run;

@end

NS_ASSUME_NONNULL_END

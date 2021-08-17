//
//  multiarray.h
//  MaskRCNN
//
//  Created by Tao Xu on 8/16/21.
//

#ifndef multiarray_h
#define multiarray_h

#import <CoreML/CoreML.h>
#include <LibTorch/LibTorch.h>

static inline at::Tensor tensorFromMultiArray(MLMultiArray* array, int64_t dim){
    std::vector<int64_t> sizes;
    int i = 0;
    for(NSNumber* n in [array.shape reverseObjectEnumerator]) {
        if( i== dim){
            break;
        }
        sizes.insert(sizes.begin(), n.intValue);
        i++;
    }
    auto tensor = at::empty(sizes, c10::kFloat);
    memcpy(tensor.data_ptr<float>(), array.dataPointer, tensor.numel() * sizeof(float));
    return tensor;
}

#endif /* multiarray_h */

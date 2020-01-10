//
//  ImageFloatBlitter.hpp
//  MNN
//
//  Created by MNN on 2018/12/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ImageFloatBlitter_hpp
#define ImageFloatBlitter_hpp

#include "ImageProcess.hpp"
#include "Tensor_generated.h"
namespace MNN {
namespace CV {
class ImageFloatBlitter {
public:
    typedef void (*BLIT_FLOAT)(const unsigned char* source, float* dest, const float* mean, const float* normal,
                               size_t count);
    // If 4 == dstBpp, use RGBA blit, otherwise use the same as format
    static BLIT_FLOAT choose(ImageFormat format, int dstBpp = 0);
};
} // namespace CV
} // namespace MNN

#endif /* ImageFloatBlitter_hpp */

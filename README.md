# CoreML-MaskRCNN

### Regarding the model

The model was trained on [d2go](https://github.com/facebookresearch/d2go). The architecture can be found at [Mask-RCNN-FBNetV3A-dsmask](https://gist.github.com/xta0/b9a0547da4e082052d704c46eb206848).

### Regarding the demo app

The demo app is located in `maskrcnn/` directory. To run the app, simply do `pod install`. For the custom ops, we're using eigen(3.3.9) for vectorization. The code was written in C++.

<img src="https://github.com/xta0/CoreML-MaskRCNN/blob/master/maskrcnn/screenshot.png" width="240">

### Run the converter

Make sure you have PyTorch installed on your machine. 

```
> import torch
> torch.__version__ # '1.9.0'
```
Run the converter
```
> python converter.py
```

### License

CoreML-MaskRCNN has a MIT-style license, as found in the LICENSE file.

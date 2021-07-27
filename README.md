# CoreML-MaskRCNN
Convert the MaskRCNN torchscript model using the coremltool.

### Before running the converter

Add the missing linear op to `{path_to_coremltools}/coremltools/converters/mil/frontend/torch/ops.py`

```python
@register_torch_op
def linear(context, node):
    x = context[node.inputs[0]]
    w = context[node.inputs[1]]
    b = context[node.inputs[2]]
    y = mb.linear(x=x, weight=w, bias=b, name=node.name)
    context.add(ssa_var=y, torch_name = node.outputs[0])
```

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


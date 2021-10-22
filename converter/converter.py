import coremltools as ct
from coremltools.converters.mil.input_types import InputType, TensorType
from coremltools.converters.mil.mil import types
from coremltools.models.neural_network import flexible_shape_utils
from coremltools.models.neural_network.builder import _get_nn_spec as get_nn
import torch
import numpy as np
import maskrcnn.custom_ops

torch_to_mil_types = {
    torch.float32: types.fp32,
    torch.float64: types.fp64,
    torch.int32: types.int32,
    torch.int64: types.int64,
}

def convert_inputs(inputs):
    input_type = []
    for index, input in enumerate(inputs):
        name = "input_" + str(index)
        ml_type = TensorType(shape=input.shape, dtype=torch_to_mil_types[input.dtype])
        ml_type.name = name
        input_type.append(ml_type)
    return input_type

m = torch.jit.load('./maskrcnn/model_freezed.pt')
graph = m._c._get_method("forward").graph
print(graph)


x1 = torch.rand(1, 3, 224, 336)
x2 = torch.rand(1, 3)
inputs = convert_inputs([x1, x2])
mlmodel = ct.convert(m, inputs=[tuple(inputs)])
print(mlmodel)
mlmodel.save("./maskrcnn_oss_coreml.mlmodel")

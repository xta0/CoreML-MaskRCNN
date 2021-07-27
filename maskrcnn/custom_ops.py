# Import MIL builder
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
# from custom_mil_ops import generateproposals
from maskrcnn.custom_mil_ops import generateproposals,roialign, bboxtransform, boxwithnmslimit
from coremltools.converters.mil.mil import types

@register_torch_op
def generateproposals(context, node):
    inputs = node.inputs
    y1, y2 = mb.generateproposals(
        scores = context[inputs[0]],
        bbox_deltas = context[inputs[1]],
        im_info = context[inputs[2]],
        anchors = context[inputs[3]],
        spatial_scale = context[inputs[4]],
        rpn_pre_nms_topN = context[inputs[5]],
        rpn_post_nms_topN = context[inputs[6]],
        nms_thresh = context[inputs[7]],
        rpn_min_size = context[inputs[8]],
        angle_bound_on = context[inputs[9]],
        angle_bound_lo = context[inputs[10]],
        angle_bound_hi = context[inputs[11]],
        clip_angle_thresh = context[inputs[12]],
        legacy_plus_one = context[inputs[13]],
        unknown=context[inputs[14]],
    )
    context.add(ssa_var=y1, torch_name=node.outputs[0])
    context.add(ssa_var=y2, torch_name=node.outputs[1])
        
@register_torch_op
def roialign(context, node):
    inputs = node.inputs
    x = mb.roialign(
        features = context[inputs[0]],
        rois = context[inputs[1]],
        order = context[inputs[2]],
        spatial_scale = context[inputs[3]],
        aligned_height = context[inputs[4]],
        aligned_width = context[inputs[5]],
        sampling_ratio = context[inputs[6]],
        aligned = context[inputs[7]],
        unknown = context[inputs[8]],
    )
    context.add(ssa_var=x, torch_name=node.outputs[0])
    
@register_torch_op
def bboxtransform(context, node):
    inputs = node.inputs    
    #  torchscript: %307 : float[] = prim::ListConstruct(%181, %181, %180, %180)
    #  MIL:   %307: (4,fp32)*(Tensor)
    y1, y2 = mb.bboxtransform(
        roi_in = context[inputs[0]],
        delta_in = context[inputs[1]],
        iminfo_in = context[inputs[2]],
        weights = context[inputs[3]], 
        apply_scale = context[inputs[4]],
        rotated = context[inputs[5]],
        angle_bound_on = context[inputs[6]],
        angle_bound_lo = context[inputs[7]],
        angle_bound_hi = context[inputs[8]],
        clip_angle_thresh = context[inputs[9]],
        legacy_plus_one = context[inputs[10]],
        unknown = context[inputs[11]],
    )
    context.add(ssa_var=y1, torch_name=node.outputs[0])
    context.add(ssa_var=y2, torch_name=node.outputs[1])
    
@register_torch_op
def boxwithnmslimit(context, node):
    inputs = node.inputs
    y1, y2, y3, y4, y5, y6 = mb.boxwithnmslimit(
        tscores = context[inputs[0]],
        tboxes = context[inputs[1]],
        tbatch_splits = context[inputs[2]],
        score_thres = context[inputs[3]],
        nms_thres = context[inputs[4]],
        detections_per_im = context[inputs[5]],
        soft_nms_enabled = context[inputs[6]],
        soft_nms_method_str = context[inputs[7]],
        soft_nms_sigma = context[inputs[8]],
        soft_nms_min_score_thres = context[inputs[9]],
        rotated = context[inputs[10]],
        cls_agnostic_bbox_reg = context[inputs[11]],
        input_boxes_include_bg_cls = context[inputs[12]],
        output_classes_include_bg_cls = context[inputs[13]],
        legacy_plus_one = context[inputs[14]],
        unknown = context[inputs[15]]
    )
    context.add(ssa_var=y1, torch_name=node.outputs[0])
    context.add(ssa_var=y2, torch_name=node.outputs[1])
    context.add(ssa_var=y3, torch_name=node.outputs[2])
    context.add(ssa_var=y4, torch_name=node.outputs[3])
    context.add(ssa_var=y5, torch_name=node.outputs[4])
    context.add(ssa_var=y6, torch_name=node.outputs[5])
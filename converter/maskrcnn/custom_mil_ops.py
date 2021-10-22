# Imports requires for custom ops
from coremltools.converters.mil.mil.ops.defs._op_reqs import *
from coremltools.converters.mil.mil.types.symbolic import is_symbolic


@register_op(doc_str="generateproposals", is_custom_op=True)
class generateproposals(Operation):
    input_spec = InputSpec(
        scores=TensorInputType(),
        bbox_deltas=TensorInputType(),
        im_info=TensorInputType(),
        anchors=TensorInputType(),
        spatial_scale=FloatInputType(const=True),
        rpn_pre_nms_topN=IntInputType(const=True),
        rpn_post_nms_topN=IntInputType(const=True),
        nms_thresh=FloatInputType(const=True),
        rpn_min_size=FloatInputType(const=True),
        angle_bound_on=BoolInputType(const=True),
        angle_bound_lo=IntInputType(const=True),
        angle_bound_hi=IntInputType(const=True),
        clip_angle_thresh=FloatInputType(const=True),
        legacy_plus_one=BoolInputType(const=True),
        unknown=ListInputType(optional=True),
    )
    bindings = {
        "class_name": "GenerateProposals",
        "input_order": ["scores", "bbox_deltas", "im_info", "anchors"],
        "parameters": [
            "spatial_scale",
            "rpn_pre_nms_topN",
            "rpn_post_nms_topN",
            "nms_thresh",
            "rpn_min_size",
            "angle_bound_on",
            "angle_bound_lo",
            "angle_bound_hi",
            "clip_angle_thresh",
            "legacy_plus_one",
            # 'unknown'
        ],
        "description": "Generate Proposals Custom Layer",
    }

    def __init__(self, **kwargs):
        super(generateproposals, self).__init__(**kwargs)

    def type_inference(self):
        print("type_inference!!!")
        dtype = self.scores.dtype
        global num_rois
        num_rois = self.rpn_post_nms_topN.val
        rois_out = [num_rois, 5]
        rois_out_prob = [num_rois]
        return types.tensor(dtype, rois_out), types.tensor(dtype, rois_out_prob)


@register_op(doc_str="roialign", is_custom_op=True)
class roialign(Operation):
    input_spec = InputSpec(
        features=TensorInputType(),
        rois=TensorInputType(),
        order=StringInputType(),
        spatial_scale=FloatInputType(),
        aligned_height=IntInputType(const=True),
        aligned_width=IntInputType(const=True),
        sampling_ratio=IntInputType(const=True),
        aligned=BoolInputType(const=True),
        unknown=ListInputType(optional=True),
    )

    bindings = {
        "class_name": "RoIAlign",
        "input_order": [
            "features",
            "rois",
        ],
        "parameters": [
            "order",
            "spatial_scale",
            "aligned_height",
            "aligned_width",
            "sampling_ratio",
            "aligned",
            # "unknown",
        ],
        "description": "Roi Align Custom Layers",
    }

    def __init__(self, **kwargs):
        super(roialign, self).__init__(**kwargs)

    def type_inference(self):
        dtype = self.features.dtype
        roi_shape = list(self.rois.shape)
        features_shape = list(self.features.shape)
        h = self.aligned_height.val
        w = self.aligned_width.val
        shape = [roi_shape[0], features_shape[1], h, w]
        return types.tensor(dtype, shape)


@register_op(doc_str="bboxtransform", is_custom_op=True)
class bboxtransform(Operation):
    input_spec = InputSpec(
        roi_in=TensorInputType(),
        delta_in=TensorInputType(),
        iminfo_in=TensorInputType(),
        weights=TensorInputType(),
        apply_scale=BoolInputType(),
        rotated=BoolInputType(),
        angle_bound_on=BoolInputType(),
        angle_bound_lo=IntInputType(),
        angle_bound_hi=IntInputType(),
        clip_angle_thresh=FloatInputType(),
        legacy_plus_one=BoolInputType(),
        unknown=ListInputType(optional=True),
    )

    bindings = {
        "class_name": "BBoxTransform",
        "input_order": [
            "roi_in",
            "delta_in",
            "iminfo_in",
            "weights",
        ],
        "parameters": [
            "apply_scale",
            "rotated",
            "angle_bound_on",
            "angle_bound_lo",
            "angle_bound_hi",
            "clip_angle_thresh",
            "legacy_plus_one",
            # "unknown",
        ],
        "description": "BBox Transformation Custom Layer",
    }

    def __init__(self, **kwargs):
        super(bboxtransform, self).__init__(**kwargs)

    def type_inference(self):
        dtype = self.roi_in.dtype
        box_out_shape = list(self.delta_in.shape)
        batch_size = self.iminfo_in.shape[0]
        roi_batch_split_shape = [batch_size]
        return types.tensor(dtype, box_out_shape), types.tensor(
            dtype, roi_batch_split_shape
        )


@register_op(doc_str="boxwithnmslimit", is_custom_op=True)
class boxwithnmslimit(Operation):
    input_spec = InputSpec(
        tscores=TensorInputType(),
        tboxes=TensorInputType(),
        tbatch_splits=TensorInputType(),
        score_thres=FloatInputType(),
        nms_thres=FloatInputType(),
        detections_per_im=IntInputType(),
        soft_nms_enabled=BoolInputType(),
        soft_nms_method_str=StringInputType(),
        soft_nms_sigma=FloatInputType(),
        soft_nms_min_score_thres=FloatInputType(),
        rotated=BoolInputType(),
        cls_agnostic_bbox_reg=BoolInputType(),
        input_boxes_include_bg_cls=BoolInputType(),
        output_classes_include_bg_cls=BoolInputType(),
        legacy_plus_one=BoolInputType(),
        unknown=ListInputType(optional=True),
    )

    bindings = {
        "class_name": "BoxWithNMSLimit",
        "input_order": [
            "tscores",
            "tboxes",
            "tbatch_splits",
        ],
        "parameters": [
            "score_thres",
            "nms_thres",
            "detections_per_im",
            "soft_nms_enabled",
            "soft_nms_method_str",
            "soft_nms_sigma",
            "soft_nms_min_score_thres",
            "rotated",
            "cls_agnostic_bbox_reg",
            "input_boxes_include_bg_cls",
            "output_classes_include_bg_cls",
            "legacy_plus_one",
            # "unknown",
        ],
        "description": "BoxWithNMSLimit Custom Layer",
    }

    def __init__(self, **kwargs):
        super(boxwithnmslimit, self).__init__(**kwargs)

    def type_inference(self):
        dtype = self.tscores.dtype
        out_scores = [num_rois]
        out_boxes = [num_rois, 4]
        out_classes = [num_rois]
        batch_split = [1]
        out_keeps = [num_rois]
        out_keeps_size = [1, 2]
        return (
            types.tensor(dtype, out_scores),
            types.tensor(dtype, out_boxes),
            types.tensor(dtype, out_classes),
            types.tensor(dtype, batch_split),
            types.tensor(dtype, out_keeps),
            types.tensor(dtype, out_keeps_size),
        )

# Imports requires for custom ops
from coremltools.converters.mil.mil.ops.defs._op_reqs import *
from coremltools.converters.mil.mil.types.symbolic import is_symbolic


# std::tuple<torch::Tensor, torch::Tensor> GenerateProposalsCPUKernel(
#     const torch::Tensor& scores,
#     const torch::Tensor& bbox_deltas,
#     const torch::Tensor& im_info_tensor,
#     const torch::Tensor& anchors_tensor,
#     double spatial_scale_,
#     int64_t rpn_pre_nms_topN_,
#     int64_t post_nms_topN_,
#     double nms_thresh_,
#     double rpn_min_size_,
#     bool angle_bound_on_,
#     int64_t angle_bound_lo_,
#     int64_t angle_bound_hi_,
#     double clip_angle_thresh_,
#     bool legacy_plus_one_,
#     c10::optional<std::vector<torch::Tensor>> /* unused */={});
# ['scores.2', 'bbox_deltas.2', 'im_info.1', '1', '172', '171', '170', '169', '168', '151', '167', '166', '165', '152', '154']
# input[0]:  %scores.1: (1, 15, 25, 14, fp32)(Tensor)
# input[1]:  %bbox_deltas.1: (1, 60, 25, 14, fp32)(Tensor)
# input[2]:  %input_1: (1, 3, fp32)(Tensor)
# input[3]:  %1: (15, 4, fp32)*(Tensor)
# input[4]:  %172: (fp32)*(Scalar)
# input[5]:  %171: (int32)*(Scalar)
# input[6]:  %170: (int32)*(Scalar)
# input[7]:  %169: (fp32)*(Scalar)
# input[8]:  %168: (fp32)*(Scalar)
# input[9]:  %151: (bool)*(Scalar)
# input[10]:  %167: (int32)*(Scalar)
# input[11]:  %166: (int32)*(Scalar)
# input[12]:  %165: (fp32)*(Scalar)
# input[13]:  %152: (bool)*(Scalar)
# input[14]:  None

@register_op(doc_str='generateproposals', is_custom_op=True)
class generateproposals(Operation):
    input_spec = InputSpec(
        scores=TensorInputType(),
        bbox_deltas=TensorInputType(),
        im_info=TensorInputType(),
        anchors=TensorInputType(const=True),
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
        unknown=ListInputType(optional=True)
    )
    bindings={
        'class_name' : 'GenerateProposals',
        'input_order': ['scores',
                        'bbox_deltas',
                        'im_info',
                        'anchors'],
        'parameters': [
            'spatial_scale',
            'rpn_pre_nms_topN',
            'rpn_post_nms_topN',
            'nms_thresh',
            'rpn_min_size',
            'angle_bound_on',
            'angle_bound_lo',
            'angle_bound_hi',
            'clip_angle_thresh',
            'legacy_plus_one',
            # 'unknown'
        ],
        'description': "Generate Proposals Custom Layer"
    }

    def __init__(self, **kwargs):
        super(generateproposals, self).__init__(**kwargs)

    def type_inference(self):
        dtype = self.scores.dtype
        # the symbolic shape it not attainable at compile time.
        return types.tensor(dtype,[]), types.tensor(dtype,[])
        
# torch::Tensor RoIAlign(
    # const torch::Tensor& features,
    # const torch::Tensor& rois,
    # std::string order,
    # double spatial_scale,
    # int64_t aligned_height,
    # int64_t aligned_width,
    # int64_t sampling_ratio,
    # bool aligned,
    # c10::optional<std::vector<torch::Tensor>>);
@register_op(doc_str='roialign', is_custom_op=True)
class roialign(Operation):
    input_spec = InputSpec(
        features = TensorInputType(),
        rois = TensorInputType(),
        order = StringInputType(),
        spatial_scale = FloatInputType(),
        aligned_height = IntInputType(const=True),
        aligned_width = IntInputType(const=True),
        sampling_ratio = IntInputType(const=True),
        aligned = BoolInputType(const=True),
        unknown = ListInputType(optional=True)
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
        "description": "Roi Align Custom Layers"
    }
    
    def __init__(self, **kwargs):
        super(roialign, self).__init__(**kwargs)

    def type_inference(self):
        dtype = self.features.dtype
        roi_shape = list(self.rois.shape)
        features_shape = list(self.features.shape)
        h = self.aligned_height.val
        w = self.aligned_width.val
        shape = [1, features_shape[1], h, w]
        # the symbolic shape it not attainable at compile time.
        return types.tensor(dtype, shape)

    
@register_op(doc_str="bboxtransform", is_custom_op=True)
class bboxtransform(Operation):
    input_spec = InputSpec(
        roi_in = TensorInputType(),
        delta_in = TensorInputType(),
        iminfo_in = TensorInputType(),
        weights = TensorInputType(),
        apply_scale = BoolInputType(),
        rotated = BoolInputType(),
        angle_bound_on = BoolInputType(),
        angle_bound_lo = IntInputType(),
        angle_bound_hi = IntInputType(),
        clip_angle_thresh = FloatInputType(),
        legacy_plus_one = BoolInputType(),
        unknown = ListInputType(optional=True)
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
        "description": "BBox Transformation Custom Layer"
    }
    
    def __init__(self, **kwargs):
        super(bboxtransform, self).__init__(**kwargs)

    def type_inference(self):
        dtype = self.roi_in.dtype
        box_out_shape = list(self.delta_in.shape)
        batch_size = self.iminfo_in.shape[0]
        roi_batch_split_shape = [batch_size]
        # the symbolic shape it not attainable at compile time.
        return types.tensor(dtype,box_out_shape), types.tensor(dtype,roi_batch_split_shape)


@register_op(doc_str="boxwithnmslimit", is_custom_op=True)
class boxwithnmslimit(Operation):
    input_spec = InputSpec(
        tscores = TensorInputType(),
        tboxes = TensorInputType(),
        tbatch_splits = TensorInputType(),
        score_thres = FloatInputType(),
        nms_thres = FloatInputType(),
        detections_per_im = IntInputType(),
        soft_nms_enabled = BoolInputType(),
        soft_nms_method_str = StringInputType(),
        soft_nms_sigma = FloatInputType(),
        soft_nms_min_score_thres = FloatInputType(),
        rotated = BoolInputType(),
        cls_agnostic_bbox_reg = BoolInputType(),
        input_boxes_include_bg_cls = BoolInputType(),
        output_classes_include_bg_cls = BoolInputType(),
        legacy_plus_one = BoolInputType(),
        unknown = ListInputType(optional=True)
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
        "description": "BoxWithNMSLimit Custom Layer"
    }
    
    def __init__(self, **kwargs):
        super(boxwithnmslimit, self).__init__(**kwargs)

    def type_inference(self):
        dtype = self.tscores.dtype
        # the symbolic shape it not attainable at compile time.
        return (
            types.tensor(dtype,[]), 
            types.tensor(dtype,[]),
            types.tensor(dtype,[]),
            types.tensor(dtype,[]),
            types.tensor(dtype,[]),
            types.tensor(dtype,[])
        )
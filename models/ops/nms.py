from models.ops import _C

from apex import amp

# Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(_C.nms)
ml_nms = amp.float_function(_C.ml_nms)

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""

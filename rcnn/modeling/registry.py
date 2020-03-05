from utils.registry import Registry


"""
Feature Extractor.
"""
# Backbone
BACKBONES = Registry()

# FPN
FPN_BODY = Registry()


"""
ROI Head.
"""
# Box Head
ROI_CLS_HEADS = Registry()
ROI_CLS_OUTPUTS = Registry()
ROI_BOX_HEADS = Registry()
ROI_BOX_OUTPUTS = Registry()

# Cascade Head
ROI_CASCADE_HEADS = Registry()
ROI_CASCADE_OUTPUTS = Registry()

# Mask Head
ROI_MASK_HEADS = Registry()
ROI_MASK_OUTPUTS = Registry()

# Keypoint Head
ROI_KEYPOINT_HEADS = Registry()
ROI_KEYPOINT_OUTPUTS = Registry()

# Parsing Head
ROI_PARSING_HEADS = Registry()
ROI_PARSING_OUTPUTS = Registry()

# UV Head
ROI_UV_HEADS = Registry()
ROI_UV_OUTPUTS = Registry()


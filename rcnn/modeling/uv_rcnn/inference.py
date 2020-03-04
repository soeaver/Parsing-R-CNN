import cv2
import numpy as np

import torch
from torch import nn

from utils.data.structures.bounding_box import BoxList
from rcnn.core.config import cfg


# TODO check if want to return a single BoxList or a composite
# object
class UVPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self):
        super(UVPostProcessor, self).__init__()

    def forward(self, uv_logits, boxes):
        """
        Arguments:
            uv_logits (List): the uv logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        UV_pred_Ann, UV_pred_Index, UV_pred_U, UV_pred_V = uv_logits

        boxes_per_image = [len(box) for box in boxes]
        UV_pred_Ann = UV_pred_Ann.split(boxes_per_image, dim=0)
        UV_pred_Index = UV_pred_Index.split(boxes_per_image, dim=0)
        UV_pred_U = UV_pred_U.split(boxes_per_image, dim=0)
        UV_pred_V = UV_pred_V.split(boxes_per_image, dim=0)

        results = []
        for Ann, Index, U, V, box in zip(UV_pred_Ann, UV_pred_Index, UV_pred_U, UV_pred_V, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("uv", [Ann.cpu().numpy(), Index.cpu().numpy(), U.cpu().numpy(), V.cpu().numpy()])
            results.append(bbox)

        return results


def uv_results(uv_logits, boxes):
    AnnIndex, Index_UV, U_uv, V_uv = uv_logits
    K = cfg.UVRCNN.NUM_PATCHES + 1
    boxes = boxes.bbox.numpy()
    uvs_results = []
    for ind, entry in enumerate(boxes):
        # Compute ref box width and height
        bx = max(entry[2] - entry[0], 1)
        by = max(entry[3] - entry[1], 1)

        # preds[ind] axes are CHW; bring p axes to WHC
        CurAnnIndex = np.swapaxes(AnnIndex[ind], 0, 2)
        CurIndex_UV = np.swapaxes(Index_UV[ind], 0, 2)
        CurU_uv = np.swapaxes(U_uv[ind], 0, 2)
        CurV_uv = np.swapaxes(V_uv[ind], 0, 2)

        # Resize p from (HEATMAP_SIZE, HEATMAP_SIZE, c) to (int(bx), int(by), c)
        CurAnnIndex = cv2.resize(CurAnnIndex, (by, bx))
        CurIndex_UV = cv2.resize(CurIndex_UV, (by, bx))
        CurU_uv = cv2.resize(CurU_uv, (by, bx))
        CurV_uv = cv2.resize(CurV_uv, (by, bx))

        # Bring Cur_Preds axes back to CHW
        CurAnnIndex = np.swapaxes(CurAnnIndex, 0, 2)
        CurIndex_UV = np.swapaxes(CurIndex_UV, 0, 2)
        CurU_uv = np.swapaxes(CurU_uv, 0, 2)
        CurV_uv = np.swapaxes(CurV_uv, 0, 2)

        # Removed squeeze calls due to singleton dimension issues
        CurAnnIndex = np.argmax(CurAnnIndex, axis=0)
        CurIndex_UV = np.argmax(CurIndex_UV, axis=0)
        CurIndex_UV = CurIndex_UV * (CurAnnIndex>0).astype(np.float32)

        output = np.zeros([3, int(by), int(bx)], dtype=np.float32)
        output[0] = CurIndex_UV

        for part_id in range(1, K):
            CurrentU = CurU_uv[part_id]
            CurrentV = CurV_uv[part_id]
            output[1, CurIndex_UV==part_id] = CurrentU[CurIndex_UV==part_id]
            output[2, CurIndex_UV==part_id] = CurrentV[CurIndex_UV==part_id]
        uvs_results.append(output)
    return uvs_results


def uv_post_processor():
    uv_post_processor = UVPostProcessor()
    return uv_post_processor

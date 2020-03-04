import torch
from torch import nn

from utils.data.structures.bounding_box import BoxList


class ParsingIoUPostProcessor(nn.Module):
    """
    Getting the parsingiou according to the targeted label, and computing the parsing score according to parsingiou.
    """
    def __init__(self):
        super(ParsingIoUPostProcessor, self).__init__()

    def forward(self, boxes, pred_parsingiou):
        num_parsings = pred_parsingiou.shape[0]
        index = torch.arange(num_parsings, device=pred_parsingiou.device)
        parsingious = pred_parsingiou[index, 0]
        parsingious = [parsingious]
        results = []
        for parsingiou, box in zip(parsingious, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox_scores = bbox.get_field("scores")
            parsing_scores = torch.sqrt(bbox_scores * parsingiou)
            bbox.add_field("parsing_scores", parsing_scores.cpu().numpy())
            prob = bbox.get_field("parsing")
            bbox.add_field("parsing", prob.cpu().numpy())
            results.append(bbox)

        return results


def parsingiou_post_processor():
    parsingiou_post_processor = ParsingIoUPostProcessor()
    return parsingiou_post_processor

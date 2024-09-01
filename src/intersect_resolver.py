from typing import Union, List
from abc import ABC, abstractmethod
import numpy as np
import torch

import box_utils.shapes_util as su
from box_utils.lines_util import find_bottom_left, find_bottom_right, find_top_right, find_top_left


def find_intersecting_objects(boxes: List[su.Obb], threshold: float = 0.4) -> List[tuple[int, int]]:
    """
    Finds interseceted boxes by IoU
    Args:
        boxes (List[Bbox, Obb]) - list of boxes
        threshold (float) - IoU threshold
    Returns
        intersecting_objects (List[tuple[int, int]]) - List of pairs containing indexes of intersected objects
    """
    intersecting_objects = []

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if su.IoU(boxes[i], boxes[j]) >= threshold:
                intersecting_objects.append((i, j))
    
    return intersecting_objects


def tensor_to_boxes(boxes: torch.Tensor) -> List[su.Obb]:
    shapes = []
    for i in range(boxes.shape[0]):
        box = boxes[i]
        shapes.append(
            su.yolo_result_to_obb(box)
        )
            
    return shapes


class IntersectionResolver(ABC):
    @abstractmethod
    def __call__(self, 
                 box_1: su.Obb, 
                 conf_1: float,
                 box_2: su.Obb,
                 conf_2: float) -> su.Obb:
        pass


class ByConfidenceResolver(IntersectionResolver):
    def __call__(self, 
                 box_1: su.Obb, 
                 conf_1: float,
                 box_2: su.Obb,
                 conf_2: float) -> su.Obb:
        return box_1 if conf_1 > conf_2 else box_2


class Unite(IntersectionResolver):
    ...


class MeanResolver(IntersectionResolver):    
    def __call__(self, 
                 box_1: su.Obb, 
                 conf_1: float,
                 box_2: su.Obb,
                 conf_2: float) -> su.Obb:
        box_1 = su.Obb(*find_top_left(box_1), 
                       *find_top_right(box_1), 
                       *find_bottom_right(box_1), 
                       *find_bottom_left(box_1))

        box_2 = su.Obb(*find_top_left(box_2), 
                       *find_top_right(box_2), 
                       *find_bottom_right(box_2), 
                       *find_bottom_left(box_2))
        
        result_box = [0] * 8
        for i in range(8):
            result_box[i] = (box_1[i] + box_2[i]) / 2
        
        return su.Obb(*result_box)


def resolve_intersected_objects(
        boxes: torch.Tensor, 
        confs: torch.Tensor,
        threshold: float, 
        resolver: IntersectionResolver
    ) -> Union[List[su.Bbox], List[su.Obb]]:
    boxes = tensor_to_boxes(boxes)
    confs = confs.to("cpu")
    intersected_boxes = find_intersecting_objects(boxes, threshold)
    
    for pair in intersected_boxes:
        i = pair[0]
        j = pair[1]
        
        chosen_box = resolver(boxes[i], confs[i], boxes[j], confs[j])
        boxes[i] = chosen_box
        boxes[j] = None
    
    return list(filter(lambda x: x != None, boxes))

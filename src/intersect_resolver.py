import box_utils.shapes_util as su
from typing import Union, List
from abc import ABC, abstractmethod
import numpy as np
import torch


def find_intersecting_objects(boxes: Union[List[su.Bbox], List[su.Obb]], threshold: float = 0.4) -> List[tuple[int, int]]:
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


def tensor_to_boxes(boxes: torch.Tensor) -> Union[List[su.Bbox], List[su.Obb]]:
    shapes = []
    for i in range(boxes.shape[0]):
        box = boxes[i]
        if len(box.shape) == 2:
            shapes.append(
                su.yolo_result_to_obb(box)
            )
        elif len(box.shape) == 1:
            shapes.append(
                su.yolo_result_to_bbox(box, su.BoxFormat.xyxy)
            )
    
    return shapes


class IntersectionResolver(ABC):
    @abstractmethod
    def __call__(self, 
                 box_1: Union[su.Bbox, su.Obb], 
                 conf_1: float,
                 box_2: Union[su.Bbox, su.Obb],
                 conf_2: float) -> Union[su.Obb, su.Bbox]:
        pass


class ChooseByConf(IntersectionResolver):
    def __call__(self, 
                 box_1: Union[su.Bbox, su.Obb], 
                 conf_1: float,
                 box_2: Union[su.Bbox, su.Obb],
                 conf_2: float) -> Union[su.Obb, su.Bbox]:
        return box_1 if conf_1 > conf_2 else box_2


class Unite(IntersectionResolver):
    ...


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
        if chosen_box == boxes[i]:
            boxes[j] = None
        elif chosen_box == boxes[j]:
            boxes[i] = None
    
    return list(filter(lambda x: x != None, boxes))

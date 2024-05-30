from math import sqrt
import numpy as np
import os

"""
TODO

Okay, I have probably figured it out. The algorithm goes as follows:

1) For each vertex of the first quadrilateral, check whether it is contained inside the second one - if so, store coordinates of the point.
2) For each vertex of the second quadrilateral, check whether it is contained inside the first one - if so, store coordinates of the point.
3) For each edge of one of the quadrilaterals (does not matter which one), check for intersections with edges of the other. Store coordinates of intersection points.
4) Compute triangulation for all the points stored so far.
5) Sum up areas of the triangles.

"""

def parse_word_bboxes(filename: str, class_num: str = "0") -> list:
    bboxes = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.split(" ")
            if values[0] != class_num:
                continue
            bboxes.append(
                list(map(float, values[1:]))
            )
    
    return bboxes


def bbox_to_obb(bbox: list) -> list:
    cx, cy = bbox[0], bbox[1]
    W, H = bbox[2], bbox[3]
    w, h = W / 2, H / 2

    return [cx - w, cy + h, cx + w, cy + h, cx + w, cy - h, cx - w, cy - h]


def is_rectangle(*coords) -> bool:
    if len(coords) != 8:
        return False

    a = np.array(coords[2:4]) - np.array(coords[:2])
    b = np.array(coords[4:6]) - np.array(coords[2:4])
    c = np.array(coords[6:8]) - np.array(coords[4:6])
    d = np.array(coords[:2]) - np.array(coords[6:8])    

    return a @ b == 0 and b @ c == 0 and c @ d == 0 and d @ a == 0


def calc_triangle_area(*coords) -> float:
    if len(coords) != 6:
        raise ValueError("Number of point coordinates must be 6")
    
    a = sqrt((coords[2] - coords[0]) ** 2 + (coords[3] - coords[1]) ** 2)
    b = sqrt((coords[4] - coords[2]) ** 2 + (coords[5] - coords[3]) ** 2)
    c = sqrt((coords[0] - coords[4]) ** 2 + (coords[1] - coords[5]) ** 2)
    p = (a + b + c) / 2

    return sqrt(p * (p - a) * (p - b) * (p - c))
    

def point_in_shape(shape: list, x: float, y: float) -> bool:
    shape_area = calc_shape_area(shape)

    point = [x, y]
    shape_points = []
    for i in range(0, len(shape) - 1, 2):
        shape_points.append([shape[i], shape[i + 1]])

    check_area = 0    
    for i in range(len(shape_points)):
        if i != len(shape_points) - 1:
            check_area += calc_triangle_area(
                *(point + shape_points[i] + shape_points[i + 1])
            )
        else:
            check_area += calc_triangle_area(
                *(point + shape_points[i] + shape_points[0])
            )

    return check_area <= shape_area


def calc_shape_area(shape: list) -> float:
    if len(shape) % 2 != 0:
        raise ValueError("The number of shape's points must be even number")
    if len(shape) <= 4:
        return 0
    
    points = []
    for i in range(0, len(shape) - 1, 2):
        points.append([shape[i], shape[i + 1]])
    
    area = 0
    for i in range(1, len(points) - 1):
        area += calc_triangle_area(*(points[0] + points[i] + points[i + 1]))
    
    return area


def intersection_shape(shape_1: list, shape_2: list) -> list:
    int_shape = []

    for i in range(0, len(shape_1) - 1, 2):
        x, y = shape_1[i], shape_1[i + 1]
        if point_in_shape(shape_2, x, y):
            int_shape.append([x, y])
    
    for i in range(0, len(shape_2) - 1, 2):
        x, y, = shape_2[i], shape_2[i + 1]
        if point_in_shape(shape_1, x, y):
            int_shape.append([x, y])

    return int_shape


def find_intersected_words(line_obb: list, word_bboxes: list, thresh: float = 0.3) -> list:
    ...


if __name__ == "__main__":
    print(
        intersection_shape([0, 0, 2, 0, 2, 2, 0, 2], [1, 1, 3, 1, 3, 3, 1, 3])
    )

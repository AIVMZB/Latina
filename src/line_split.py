from math import sqrt
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

"""
TODO

Okay, I have probably figured it out. The algorithm goes as follows:

1) For each vertex of the first quadrilateral, check whether it is contained inside the second one - if so, store coordinates of the point.
2) For each vertex of the second quadrilateral, check whether it is contained inside the first one - if so, store coordinates of the point.
3) For each edge of one of the quadrilaterals (does not matter which one), check for intersections with edges of the other. Store coordinates of intersection points.
4) Compute triangulation for all the points stored so far.
5) Sum up areas of the triangles.

"""

def read_bboxes(filename: str, class_num: str = "0") -> list:
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

def obb_to_polygon(obb: list) -> Polygon:
    points = []
    for i in range(0, len(obb), 2):
        points.append((obb[i], obb[i + 1]))
    
    return Polygon(points)

def intersection_area(shape_1: list, shape_2: list) -> list:
    pol_1 = obb_to_polygon(shape_1)
    pol_2 = obb_to_polygon(shape_2)

    return pol_1.intersection(pol_2).area

def obb_to_image_coords(width: int, height: int, obb: list) -> list:
    image_obb = []
    for i in range(len(obb)):
        if i % 2 == 0:
            image_obb.append(int(obb[i] * width))
        else:
            image_obb.append(int(obb[i] * height))

    return image_obb

def find_intersected_words(line_obb: list, word_bboxes: list, 
                           width: int, height: int, 
                           thresh: float = 0.3) -> list:
    """
    line_obb - bbox of line
    word_bboxes - list of bboxes of words
    thresh - intersection threshold
    """
    
    intersected_words = []
    line_obb = obb_to_image_coords(width, height, line_obb)
    for word_bbox in word_bboxes:
        word_obb = bbox_to_obb(word_bbox)
        word_obb = obb_to_image_coords(width, height, word_obb)
        
        if intersection_area(line_obb, word_obb) / obb_to_polygon(word_obb).area > thresh:
            intersected_words.append(word_bbox)

    return intersected_words

def show_obb_on_image(image: np.ndarray, obb: list, color: tuple = (0, 0, 255)) -> np.ndarray:
    im_obb = np.zeros((4, 2), dtype=np.int32)
    width = image.shape[0]
    height = image.shape[1]

    for i in range(0, len(obb), 2):
        im_obb[i // 2][0] = round(obb[i] * height) if obb[i] > 0 else 0
        im_obb[i // 2][1] = round(obb[i + 1] * width) if obb[i + 1] > 0 else 0
    
    im_obb = im_obb.reshape((-1, 1, 2))

    return cv2.polylines(image, [im_obb], True, color, 5)


def show_bbox_on_image(image: np.ndarray, bbox: list, color: tuple = (0, 0, 255)) -> np.ndarray:
    obb = bbox_to_obb(bbox)

    return show_obb_on_image(image, obb, color)

if __name__ == "__main__":
    lines_obb_data = "../datasets/lines-obb"
    words_data = "../datasets/words"
    
    lines = read_bboxes("..\datasets\lines-obb\\train\labels\AUR_1051_II_08-101-text-_jpg.rf.0d97b8261dce428a4f663c3ffef3e02c.txt")
    words = read_bboxes("..\datasets\words\\train\AUR_1051_II_08-101 (text).txt")

    # words_in_line = find_intersected_words(line, words)
    for line in lines:
        image = cv2.imread("..\datasets\lines-obb\\train\images\AUR_1051_II_08-101-text-_jpg.rf.0d97b8261dce428a4f663c3ffef3e02c.jpg")

        int_words = find_intersected_words(line, words, image.shape[1], image.shape[0], thresh=0.5)

        for word in words:
            image = show_bbox_on_image(image, word)
        
        for word in int_words:
            image = show_bbox_on_image(image, word, (255, 0, 0))

        image = show_obb_on_image(image, line, (0, 255, 0))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.show()

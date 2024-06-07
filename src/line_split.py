import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from typing import NamedTuple, Union, Sequence
from prettyprinter import pprint


class Bbox(NamedTuple):
    cx: float
    cy: float
    w: float
    h: float


class Obb(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float


def to_bbox(bbox_list: Sequence) -> Bbox:
    """
    Creates a Bbox object from a sequence of values.

    Args:
        bbox_list (Sequence): A sequence of values representing the coordinates of a bounding box. The sequence should have exactly four values: [x, y, width, height].

    Returns:
        Bbox: A Bbox object representing the bounding box with the given coordinates.
    """

    return Bbox(*bbox_list)


def to_obb(shape: Union[Bbox, Sequence]) -> Obb:
    """
    Convert a shape to an Obb (Oriented Bounding Box) object.

    Args:
        shape (Union[Bbox, Sequence]): The shape to be converted. It can be either a Bbox object or a sequence of values.

    Returns:
        Obb: The Obb object representing the converted shape.

    Raises:
        TypeError: If the shape is neither a Bbox object nor a sequence of values.

    """
    if isinstance(shape, Bbox):
        return bbox_to_obb(shape)
    elif isinstance(shape, Sequence):
        return Obb(*shape)


def read_shapes(filename: str, class_num: str = "0", 
                transform_func: callable = to_bbox) -> Union[list[Bbox], list[Obb]]:
    """
    Reads shapes from a file and transforms them using a given function.

    Args:
        filename (str): The name of the file to read.
        class_num (str, optional): The class number to filter the shapes. Defaults to "0".
        transform_func (callable, optional): The function to transform the shapes. Defaults to to_bbox.

    Returns:
        Union[list[Bbox], list[Obb]]: A list of readed shapes.
    """
    shapes = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.split(" ")
            if values[0] != class_num:
                continue
            shapes.append(
                transform_func(list(map(float, values[1:])))
            )
    
    return shapes


def bbox_to_obb(bbox: Union[list, Bbox]) -> Obb:
    cx, cy = bbox[0], bbox[1]
    W, H = bbox[2], bbox[3]
    w, h = W / 2, H / 2

    return to_obb([cx - w, cy + h, cx + w, cy + h, cx + w, cy - h, cx - w, cy - h])


def obb_to_polygon(obb: Union[list, Obb]) -> Polygon:
    points = []
    for i in range(0, len(obb), 2):
        points.append((obb[i], obb[i + 1]))

    return Polygon(points)


def intersection_area(shape_1: Obb, shape_2: Obb) -> list:
    pol_1 = obb_to_polygon(shape_1)
    pol_2 = obb_to_polygon(shape_2)

    return pol_1.intersection(pol_2).area


def obb_to_image_coords(width: int, height: int, obb: Obb) -> Obb:
    image_obb = []
    for i in range(len(obb)):
        if i % 2 == 0:
            image_obb.append(int(obb[i] * width))
        else:
            image_obb.append(int(obb[i] * height))

    return to_obb(image_obb)


def find_line_for_word(word_obb: Obb, line_bboxes: list[Obb]) -> Obb:
    """Counts intersection of lines with the specified word and picks a line with the highest intersection area"""

    best_intersection = 0
    best_line_index = None
    for i, line_bbox in enumerate(line_bboxes):
        intersection = intersection_area(word_obb, line_bbox)
        if intersection > best_intersection:
            best_intersection = intersection
            best_line_index = i
            
    return best_line_index


def find_words_in_line(line_obb: Obb, word_bboxes: list[Bbox], 
                       width: int, height: int, 
                       thresh: float = 0.5) -> list:
    """
    Finds the indexes of words that intersect with a given line.

    Args:
        line_obb (Obb): The oriented bounding box of the line.
        word_bboxes (list[Bbox]): The bounding boxes of the words.
        width (int): The width of the image.
        height (int): The height of the image.
        thresh (float, optional): The threshold for intersection area. Defaults to 0.5.

    Returns:
        list: The indexes of the intersected words.
    """
    
    intersected_words_indexes = []
    line_obb = obb_to_image_coords(width, height, line_obb)
    for i, word_bbox in enumerate(word_bboxes):
        word_obb = bbox_to_obb(word_bbox)
        word_obb = obb_to_image_coords(width, height, word_obb)
        
        if intersection_area(line_obb, word_obb) / obb_to_polygon(word_obb).area > thresh:
            intersected_words_indexes.append(i)

    return intersected_words_indexes


def show_obb_on_image(image: np.ndarray, obb: Obb, color: tuple = (0, 0, 255)) -> np.ndarray:
    im_obb = np.zeros((4, 2), dtype=np.int32)
    width = image.shape[0]
    height = image.shape[1]

    for i in range(0, len(obb), 2):
        im_obb[i // 2][0] = round(obb[i] * height) if obb[i] > 0 else 0
        im_obb[i // 2][1] = round(obb[i + 1] * width) if obb[i + 1] > 0 else 0
    
    im_obb = im_obb.reshape((-1, 1, 2))

    return cv2.polylines(image, [im_obb], True, color, 5)


def show_bbox_on_image(image: np.ndarray, bbox: Bbox, color: tuple = (0, 0, 255)) -> np.ndarray:
    obb = bbox_to_obb(bbox)

    return show_obb_on_image(image, obb, color)


def show_word_and_line(image: np.ndarray, word: Bbox, lines: list[Obb]) -> np.ndarray:
    line = find_line_for_word(word, lines)

    image = show_obb_on_image(image, line)
    return show_obb_on_image(image, word, (255, 0, 0))


def index_in_map(table: dict, index: int) -> bool:
    for values in table.values():
        if index in values:
            return True

    return False


def map_words_to_lines(words: list[Bbox], lines: list[Obb], image: np.ndarray) -> dict:
    line_to_words = {}
    count = 0
    for i, line in enumerate(lines):
        line_to_words[i] = find_words_in_line(line, words, image.shape[1], image.shape[0])
        count += len(line_to_words[i])
    
    if count == len(words):
        return line_to_words
    
    for i, word in enumerate(words):
        if index_in_map(line_to_words, i):
            continue
        
        line_index = find_line_for_word(to_obb(word), lines)

        line_to_words[line_index].append(i)

    return line_to_words


if __name__ == "__main__":
    lines = read_shapes("..\datasets\lines-obb\\test\labels\AUR_945_VI_4-101-text-_jpg.rf.f05201b45f7215a2eb52c9750eed34e6.txt",
                        transform_func=to_obb)
    words = read_shapes("..\datasets\words\\train\AUR_945_VI_4-101 (text).txt",
                        transform_func=to_bbox)

    image = cv2.imread("../datasets/lines-obb/test/images/AUR_945_VI_4-101-text-_jpg.rf.f05201b45f7215a2eb52c9750eed34e6.jpg")

    line_to_words = map_words_to_lines(words, lines, image)
    pprint(line_to_words)

    # for i in line_to_words[0]:
    #     word = words[i]
    #     word = to_obb(word)
    #     # word = obb_to_image_coords(image.shape[1], image.shape[0], word)
    #     image = show_obb_on_image(image, word)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show()

from shapes_util import Bbox, Obb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shapes_util as su
from skimage.draw import polygon2mask


# TODO: Implement transforming words coordinates to line space according to the belonging

def obb_to_nparray(obb: Obb) -> np.ndarray:
    points = []
    for i in range(0, len(obb), 2):
        points.append([obb[i], obb[i + 1]])

    return np.array(points)


def obb_to_bbox(line: Obb) -> Bbox:
    points = obb_to_nparray(line)
    x, y, w, h = cv2.boundingRect(points)
    x = (x + w) / 2
    y = (y + h) / 2
    return Bbox(x, y, w, h)


def obb_to_rec(obb: Obb) -> list:
    """Returns x, y, w, h"""
    x = min(obb.x1, obb.x2, obb.x3, obb.x4)
    y = min(obb.y1, obb.y2, obb.y3, obb.y4)
    w = max(obb.x1, obb.x2, obb.x3, obb.x4) - x
    h = max(obb.y1, obb.y2, obb.y3, obb.y4) - y

    return [x, y, w, h]


def crop_line_from_image(image: np.ndarray, line: Obb, white_bg: bool = False) -> np.ndarray:
    line = su.obb_to_image_coords(image.shape[1], image.shape[0], line)
    np_line = obb_to_nparray(line)
    for i in range(np_line.shape[0]):
        np_line[i][0], np_line[i][1] = np_line[i][1], np_line[i][0]

    line_mask = polygon2mask(image.shape[:2], np_line)

    line_mask = np.stack([line_mask, line_mask, line_mask], axis=-1)

    masked_image = image * line_mask
    if white_bg:
        masked_image = np.vectorize(lambda x: 255 if x == 0 else x)(masked_image)

    x, y, w, h = obb_to_rec(line)

    croped = masked_image[y:y+h, x:x+w].copy()

    return croped


def bbox_to_line_space(bbox: Bbox, line: list) -> Bbox:
    """
    Transforms the bounding box coordinates to line space according to the provided line.

    Args:
        bbox (Bbox): The bounding box coordinates.
        line (list): The line in form of rectangle (x, y, w, h).

    Returns:
        Bbox: The transformed bounding box in line space.
    """
    bbox_center = np.array([bbox.cx, bbox.cy])
    bbox_size = np.array([bbox.w, bbox.h])
    line_point = np.array([line[0], line[1]])
    line_size = np.array([line[2], line[3]])

    line_bbox_center = (bbox_center - line_point) / line_size
    line_bbox_size = bbox_size / line_size

    bbox = Bbox(*line_bbox_center, *line_bbox_size)

    # if bbox.cx + bbox.w / 2 > 1:
    #     bbox = Bbox(bbox.cx, bbox.cy, 2 * (1 - bbox.cx), bbox.h)
    # if bbox.cx - bbox.w / 2 < 0:
    #     bbox = Bbox(bbox.cx, bbox.cy, 2 * bbox.cx, bbox.h)

    # if bbox.cy + bbox.h / 2 > 1:
    #     bbox = Bbox(bbox.cx, bbox.cy, bbox.w, 2 * (1 - bbox.cy))
    # if bbox.cy - bbox.h / 2 < 0:
    #     bbox = Bbox(bbox.cx, bbox.cy, bbox.w, 2 * bbox.cy)

    return bbox


def words_to_line_space(words: list[Bbox], line: Obb) -> list[Bbox]:
    line_words = []
    line_rec = obb_to_rec(line)
    for word in words:
        word_in_line = bbox_to_line_space(word, line_rec)
        line_words.append(word_in_line)

    return line_words


if __name__ == "__main__":
    # TODO: Create datased with image for each
    image = cv2.imread(
        "..\datasets\lines-obb\\train\images\AUR_996_V_28-101-text-_jpg.rf.51ea5ad9e77ebc31d1c6c1b66cf5adb1.jpg")
    lines: list[Obb] = su.read_shapes(
        "..\datasets\lines-obb\\train\labels\AUR_996_V_28-101-text-_jpg.rf.51ea5ad9e77ebc31d1c6c1b66cf5adb1.txt",
        transform_func=su.to_obb)
    words = su.read_shapes("..\datasets\words\\train\AUR_996_V_28-101 (text).txt", transform_func=su.to_bbox)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    line_to_words = su.map_words_to_lines(words, lines, image)

    for line_idx in line_to_words:
        line = lines[line_idx]        
        im_line = crop_line_from_image(image, line, white_bg=True)

        line_words = [words[index] for index in line_to_words[line_idx]]
        line_words = words_to_line_space(line_words, line)
        for word in line_words:
            im_line = su.plot_obb_on_image(im_line, su.to_obb(word))

        plt.imshow(im_line)
        plt.show()


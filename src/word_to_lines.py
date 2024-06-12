from shapes_util import Bbox, Obb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shapes_util as su

# TODO: Implement transforming words coordinates to line space according to the belonging

def transform_words_to_line_space(words: list[Bbox], line: Obb, image: np.ndarray) -> list[Bbox]:
    """
    The functions transforms words coordinate according to the line, which the words belong to.

    Args:
        words (list[Bbox]): The list of words.
        line (Obb): The oriented bounding box of the line.
        image (np.ndarray): The image.

    Returns:
        list[Bbox]: The list of words in line space.
    """
    ...

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

def obb_to_rec(obb: Obb, im_w: int = 0, im_h: int = 0) -> list:
    "Returns x, y, w, h"
    if im_w != 0 or im_h != 0:
        obb = su.obb_to_image_coords(im_w, im_h, obb)

    points = obb_to_nparray(obb)
    x, y, w, h = cv2.boundingRect(points)
    
    if im_w != 0 or im_h != 0:
        return [x / im_w, y / im_h, w / im_w, h / im_h]
    return [x, y, w, h]

def crop_line_from_image(image: np.ndarray, line: Obb) -> np.ndarray:
    line = su.obb_to_image_coords(image.shape[1], image.shape[0], line)
    x, y, w, h = obb_to_rec(line)
    croped = image[y:y+h, x:x+w].copy()
    
    # Maybe try to add masking 

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
    ...

if __name__ == "__main__":
    # TODO: Create datased with image for each
    image = cv2.imread("../datasets/lines-obb/test/images/AUR_945_VI_4-101-text-_jpg.rf.f05201b45f7215a2eb52c9750eed34e6.jpg")
    lines = su.read_shapes("..\datasets\lines-obb\\test\labels\AUR_945_VI_4-101-text-_jpg.rf.f05201b45f7215a2eb52c9750eed34e6.txt",
                        transform_func=su.to_obb)
    words = su.read_shapes("..\datasets\words\\train\AUR_945_VI_4-101 (text).txt", transform_func=su.to_bbox)
    
    line_to_words = su.map_words_to_lines(words, lines, image)

    for line_idx in line_to_words:
        line = lines[line_idx]        
        im_line = crop_line_from_image(image, line)

        for word_idx in line_to_words[line_idx]:
            word = words[word_idx]

            line_rec = obb_to_rec(line, im_line.shape[1], im_line.shape[0])
            word_in_line = bbox_to_line_space(word, line_rec)

            im_line = su.plot_obb_on_image(im_line, su.to_obb(word_in_line))

        plt.imshow(im_line)
        plt.show()
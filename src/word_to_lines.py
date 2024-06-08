from shapes_util import Bbox, Obb
import numpy as np


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



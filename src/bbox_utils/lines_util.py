from shapes_util import Obb, read_shapes, to_obb, plot_obb_on_image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from prettyprinter import pprint
from ultralytics import YOLO


def extend_lines_to_corners(lines: list[Obb]) -> list[Obb]:
    extended_lines = []
    for line in lines:
        extended_lines.append(
            extend_line_to_corners(line)
        )

    return extended_lines


def extend_line_to_corners(line: Obb) -> Obb:
    top_direction = np.array([line.x2 - line.x3, line.y2 - line.y3])
    bottom_direction = np.array([line.x1 - line.x4, line.y1 - line.y4])
    avg_direction = (top_direction + bottom_direction) / 2

    x3 = x4 = 0
    x2 = x1 = 1
    y3 = line.y3 - line.x3 * avg_direction[1] / avg_direction[0]
    y4 = line.y4 - line.x4 * avg_direction[1] / avg_direction[0]

    y1 = line.y1 + avg_direction[1] * (1 - line.x1) / avg_direction[0]
    y2 = line.y2 + avg_direction[1] * (1 - line.x2) / avg_direction[0]

    return Obb(x1, y1, x2, y2, x3, y3, x4, y4)


def save_model_results(results: np.ndarray) -> None:
    with open("results.txt", 'w') as f:
        for i in range(results.shape[0]):
            box = results[i]
            f.write(f"{box[0, 0]} {box[0, 1]} {box[1, 0]} {box[1, 1]} {box[2, 0]} {box[2, 1]} {box[3, 0]} {box[3, 1]}\n")


def obbs_from_file(filename: str = "results.txt") -> list[Obb]:
    obbs = []
    with open(filename, 'r') as f:
        for line in f:
            values = [float(value) for value in line.split(" ")]
            obbs.append(Obb(*values))
    
    return obbs


def model_results_to_obbs(results: np.ndarray) -> list[Obb]:
    obbs = []
    for i in range(results.shape[0]):
        box = results[i]
        obbs.append(
            Obb(box[0, 0], box[0, 1], 
                box[1, 0], box[1, 1], 
                box[2, 0], box[2, 1], 
                box[3, 0], box[3, 1])
        )
    
    return obbs

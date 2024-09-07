from shapes import Obb
import numpy as np


def find_top_left(box: Obb) -> tuple[float, float]:
    left_1 = 0
    for i in range(2, 8, 2):
        if box[i] < box[left_1]:
            left_1 = i
        
    left_2 = 0 if left_1 != 0 else 2
    for i in range(0, 8, 2):
        if box[i] < box[left_2] and i != left_1:
            left_2 = i

    if box[left_1 + 1] < box[left_2 + 1]:
        return box[left_1], box[left_1 + 1]
        
    return box[left_2], box[left_2 + 1]


def find_bottom_left(box: Obb) -> tuple[float, float]:
    left_1 = 0
    for i in range(0, 8, 2):
        if box[i] < box[left_1]:
            left_1 = i
        
    left_2 = 0 if left_1 != 0 else 2
    for i in range(0, 8, 2):
        if box[i] < box[left_2] and i != left_1:
            left_2 = i

    if box[left_1 + 1] > box[left_2 + 1]:
        return box[left_1], box[left_1 + 1]
        
    return box[left_2], box[left_2 + 1]
    

def find_top_right(box: Obb) -> tuple[int, int]:
    right_1 = 0
    
    for i in range(2, 8, 2):
        if box[i] > box[right_1]:
            right_1 = i

    right_2 = 0 if right_1 != 0 else 2    
    for i in range(0, 8, 2):
        if box[i] > box[right_2] and i != right_1:
            right_2 = i

    if box[right_1 + 1] < box[right_2 + 1]:
       return box[right_1], box[right_1 + 1]

    return box[right_2], box[right_2 + 1]
    

def find_bottom_right(box: Obb) -> tuple[int, int]:
    right_1 = 0

    for i in range(2, 8, 2):
        if box[i] > box[right_1]:
            right_1 = i

    right_2 = 0 if right_1 != 0 else 2        
    for i in range(0, 8, 2):
        if box[i] > box[right_2] and i != right_1:
            right_2 = i

    if box[right_1 + 1] > box[right_2 + 1]:
        return box[right_1], box[right_1 + 1]

    return box[right_2], box[right_2 + 1]


def extend_lines_to_corners(lines: list[Obb]) -> list[Obb]:
    extended_lines = []
    for line in lines:
        extended_lines.append(
            extend_line_to_corners(line)
        )

    return extended_lines


def extend_line_to_corners(line: Obb) -> Obb:
    top_left = find_top_left(line)
    top_right = find_top_right(line)
    bottom_left = find_bottom_left(line)
    bottom_right = find_bottom_right(line)
    
    top_direction = np.array([top_right[0] - top_left[0], top_right[1] - top_left[1]])
    bottom_direction = np.array([bottom_right[0] - bottom_left[0], bottom_right[1] - bottom_left[1]])
    avg_direction = (top_direction + bottom_direction) / 2

    x3 = x4 = 0
    x2 = x1 = 1
    y3 = bottom_left[1] - bottom_left[0] * avg_direction[1] / avg_direction[0]
    y4 = top_left[1] - top_left[0] * avg_direction[1] / avg_direction[0]

    y1 = top_right[1] + avg_direction[1] * (1 - top_right[0]) / avg_direction[0]
    y2 = bottom_right[1] + avg_direction[1] * (1 - bottom_right[0]) / avg_direction[0]

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


def line_angle(line: Obb) -> float:
    top_left = find_top_left(line)
    top_right = find_top_right(line)
    bottom_left = find_bottom_left(line)
    bottom_right = find_bottom_right(line)
    
    top_direction = np.array([top_right[0] - top_left[0], top_right[1] - top_left[1]])
    bottom_direction = np.array([bottom_right[0] - bottom_left[0], bottom_right[1] - bottom_left[1]])
    avg_direction = (top_direction + bottom_direction) / 2
    avg_direction = avg_direction / np.sqrt((avg_direction @ avg_direction.T))

    radians = np.arctan(avg_direction[1] / avg_direction[0])
    
    return np.degrees(radians)

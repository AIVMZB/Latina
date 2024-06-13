from shapes_util import Bbox, Obb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shapes_util as su
from skimage.draw import polygon2mask
import os
import random
import shutil


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

def write_bboxes_to_file(bboxes: list[Bbox], filename: str, class_name: str) -> None:
    with open(filename, 'w') as file:
        for bbox in bboxes:
            line = class_name + " " + " ".join(map(str, bbox)) + "\n"
            file.write(line)


def create_lines_dataset_from_image(image_name: str, image_dir: str, 
                                    lines: list[Obb], words: list[Bbox], output_dir: str, 
                                    word_class: str = '0',
                                    white_bg: bool = False) -> None:
    if not os.path.exists(os.path.join(image_dir, image_name)):
        raise ValueError(f"There is no such an image {os.path.join(image_dir, image_name)}")
    
    image = cv2.imread(os.path.join(image_dir, image_name))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    line_to_words = su.map_words_to_lines(words, lines, image)
    for line_idx in line_to_words:
        line = lines[line_idx]

        line_image = crop_line_from_image(image, line, white_bg=white_bg)
        line_image_path = os.path.join(output_dir, f"{image_name[:-4]}_{line_idx}.jpg")

        try:
            cv2.imwrite(line_image_path, line_image)
        except:
            print(f"\t[WARNING] Failed to save line image with index {line_idx}")
        
        line_words = [words[idx] for idx in line_to_words[line_idx]]
        line_words = words_to_line_space(line_words, line)

        write_bboxes_to_file(line_words, os.path.join(output_dir, f"{image_name[:-4]}_{line_idx}.txt"), word_class)
    

def find_label_file_in_dirs(dirs: list[str], image_name: str) -> str:
    for dir in dirs:
        for file in os.listdir(dir):
            if file.startswith(image_name[:-12]) and file.endswith(".txt"):
                return os.path.join(dir, file)

def create_dataset_of_lines(words_dataset_path: str = "..\datasets\words",
                            lines_dataset_path: str = "..\datasets\lines-obb",
                            output_dataset_path: str = "..\datasets\words-in-lines"):
    word_train_dir = os.path.join(words_dataset_path, "train")
    word_val_dir = os.path.join(words_dataset_path, "validation")
    
    lines_train_dir = os.path.join(lines_dataset_path, "train", "labels")
    lines_test_dir = os.path.join(lines_dataset_path, "test", "labels")
    lines_val_dir = os.path.join(lines_dataset_path, "valid", "labels")
    
    print("------------------ Train ------------------")
    for word_image_name in os.listdir(word_train_dir)[1:]:
        if not word_image_name.endswith(".jpg"):
            continue

        word_label_file = os.path.join(word_train_dir, word_image_name[:-4] + ".txt")
        line_label_file = find_label_file_in_dirs([lines_train_dir, lines_test_dir, lines_val_dir], word_image_name)
        if line_label_file is None:
            print(f"[WARNING] {word_image_name} is not found in lines dataset")
            continue
        
        lines = su.read_shapes(line_label_file, transform_func=su.to_obb)
        words = su.read_shapes(word_label_file, transform_func=su.to_bbox)

        print(f"[INFO] Processing {word_image_name}")
        try:
            create_lines_dataset_from_image(word_image_name, word_train_dir, lines, words, output_dataset_path, white_bg=False)
        except Exception as e:
            print(f"[ERROR] {word_image_name} failed")
            print(e)

    ######################################
    print("------------------ Validation ------------------")
    for word_image_name in os.listdir(word_val_dir):
        if not word_image_name.endswith(".jpg"):
            continue

        word_label_file = os.path.join(word_val_dir, word_image_name[:-4] + ".txt")
        line_label_file = find_label_file_in_dirs([lines_train_dir, lines_test_dir, lines_val_dir], word_image_name)
        if line_label_file is None:
            print(f"[WARNING] {word_image_name} is not found in lines dataset")
            continue
        
        lines = su.read_shapes(line_label_file, transform_func=su.to_obb)
        words = su.read_shapes(word_label_file, transform_func=su.to_bbox)

        print(f"[INFO] Processing {word_image_name}")
        try:
            create_lines_dataset_from_image(word_image_name, word_val_dir, lines, words, output_dataset_path, white_bg=False)
        except Exception as e:
            print(f"[ERROR] {word_image_name} failed")
            print(e)

def split_dataset(dataset_path: str, output_path: str):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    if not os.path.exists(os.path.join(output_path, "train")):
        os.mkdir(os.path.join(output_path, "train"))
    
    if not os.path.exists(os.path.join(output_path, "validation")):
        os.mkdir(os.path.join(output_path, "validation"))
    
    for file in os.listdir(dataset_path):
        if file.endswith(".jpg"):
            if random.random() < 0.8:
                shutil.copy(os.path.join(dataset_path, file), os.path.join(output_path, "train", file))
                shutil.copy(os.path.join(dataset_path, file[:-4] + ".txt"), os.path.join(output_path, "train", file[:-4] + ".txt"))
            else:
                shutil.copy(os.path.join(dataset_path, file), os.path.join(output_path, "validation", file))
                shutil.copy(os.path.join(dataset_path, file[:-4] + ".txt"), os.path.join(output_path, "validation", file[:-4] + ".txt"))


if __name__ == "__main__":
    # TODO: split to train and validation

    # Creation of lines dataset
    # create_dataset_of_lines()
    # image_name = "AUR_889_X_5(15)-101 (text).jpg"
    # image_dir = "..\\datasets\\words\\validation"
    # lines = su.read_shapes("..\datasets\lines-obb\\test\labels\AUR_889_X_5-15-101-text-_jpg.rf.e2800ba0157f1988cbd011f68c072622.txt",
    #                         transform_func=su.to_obb)
    # words = su.read_shapes("..\datasets\words\\validation\AUR_889_X_5(15)-101 (text).txt", 
    #                        transform_func=su.to_bbox)
    # create_lines_dataset_from_image(image_name, image_dir, lines, words, "..\datasets\words-in-lines", white_bg=False)

    # Splitting it to train and validation
    # split_dataset("..\datasets\words-in-lines", "..\datasets\words-in-lines-splitted")

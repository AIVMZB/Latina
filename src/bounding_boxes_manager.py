from os import path
import matplotlib.pyplot as plt
import matplotlib.style

"""
Requires matplotlib module
Input file is a text file with YOLOv8 bounding box format

Yellow number in visualization - original order of bounding box
White number in visualization - sorted order of bounding box 
"""

class BoundingBox:
    def __init__(self, class_id: int, x: float, y: float, w: float, h: float):
        self.class_id = class_id
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        return f"Class ID: {self.class_id}, X: {self.x}, Y: {self.y}, W: {self.w}, H: {self.h}"


class BoundingBoxManager:
    def __init__(self, input_file: str) -> None:
        self.bounding_boxes_sorted: list = []
        self.bounding_boxes: list = []

        self._parse_file(input_file)
    
    def sort(self) -> None:
        """ Needs improvements """
        self.boxes_by_line: dict = {}

        line_height = format(
            max([bounding_box.y + bounding_box.h for bounding_box in self.bounding_boxes]),
            ".2f"
        )

        for bounding_box in self.bounding_boxes:
            line_id = round(bounding_box.y / float(line_height), 1)
            self.boxes_by_line[line_id] = []
        
        for bounding_box in self.bounding_boxes:
            line_id = round(bounding_box.y / float(line_height), 1)
            self.boxes_by_line[line_id].append(bounding_box)

        self.bounding_boxes_sorted_by_line = [
            sorted(box_list, key=lambda box: box.x)
            for line_id, box_list in self.boxes_by_line.items()
        ]
        
        self.bounding_boxes_sorted = [
            box
            for line_list in self.bounding_boxes_sorted_by_line
            for box in line_list
        ]
        
        self.bounding_boxes_sorted = [
            bounding_box
            for line_id, box_list in self.boxes_by_line.items()
            for bounding_box in box_list
        ]

    def visualize(self) -> None:
        matplotlib.style.use("dark_background")
        fig, ax = plt.subplots()
        
        for i, box in enumerate(self.bounding_boxes):
            plt.text(box.x + box.w/2, box.y + box.h/2, f"{i}", ha="center", va="center", color="gold")
        
        for i, box in enumerate(self.bounding_boxes_sorted):
            rect = plt.Rectangle((box.x, box.y), box.w, box.h, color="springgreen", alpha=0.5)
            ax.add_patch(rect)
            plt.text(box.x + box.w/2, box.y + box.h/2 + 0.01, f"{i}", ha="center", va="center", color="azure")
        
        plt.axis("scaled")
        plt.gca().invert_yaxis()
        plt.show()

    def _parse_file(self, file_path):
        if not path.isfile(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist.")

        with open(file_path, 'r') as file:
            for line in file:
                values = line.strip().split()

                if len(values) != 5:
                    raise ValueError(f"Invalid line format in file '{file_path}': {line}")

                class_id = int(values[0])
                x, y, w, h = map(float, values[1:6])
                bounding_box = BoundingBox(class_id, x, y, w, h)
                self.bounding_boxes.append(bounding_box)


if __name__ == "__main__":
    boxctl = BoundingBoxManager("labels.txt")
    boxctl.sort()
    boxctl.visualize()
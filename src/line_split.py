"""
Okay, I have probably figured it out. The algorithm goes as follows:

For each vertex of the first quadrilateral, check whether it is contained inside the second one - if so, store coordinates of the point.
For each vertex of the second quadrilateral, check whether it is contained inside the first one - if so, store coordinates of the point.
For each edge of one of the quadrilaterals (does not matter which one), check for intersections with edges of the other. Store coordinates of intersection points.
Compute triangulation for all the points stored so far.
Sum up areas of the triangles.
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
    w, h = bbox[2], bbox[3]


def find_intersected_words(line_obb: list, word_bboxes: list, thresh: float = 0.3) -> list:
    ...

if __name__ == "__main__":
    bboxes = parse_word_bboxes("..\\datasets\\lines-obb\\train\\labels\\AUR_816_II_5-101-text-_jpg.rf.cd88b489d982326041cbf2c2fe28dec8 – копія.txt", "0")
    print(bboxes)

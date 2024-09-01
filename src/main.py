import os
from model_wrapper import LineWordPipeline


WORD_DETECTION_IMG_SIZE = 512
LINE_DETECTION_IMG_SIZE = 768

WORD_DETECT_BEST_MODEL = os.path.join("..", "models", "word_detect_m_best.pt")
LINES_OBB_BEST = os.path.join("..", "models", "lines_obb_m_best.pt")

# TODO:
# 1) Deal with intersecting words and lines
#   a) Fix Mean Resolver
#   b) Create UnionResolver
# 2) Move plot function to separate module
# 3) Rename shapes_util to shapes

if __name__ == "__main__":
    # image_path = input("Input image path here\n>>> ")
    image_path = r"E:\Dyploma\Latina\LatinaProject\images\AUR_1020_IV_23-101 (text).jpg"

    pipeline = LineWordPipeline(LINES_OBB_BEST, WORD_DETECT_BEST_MODEL, "cuda:0")
    pipeline.predict_on_image(
        image_path,
        r"../predictions",
        line_conf=0.35
    )

    print("The results are saved in '../predictions' folder")


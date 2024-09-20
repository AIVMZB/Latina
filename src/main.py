import os
import yaml
import argparse

from detection.model_wrapper import build_line_word_pipeline


WORD_DETECTION_IMG_SIZE = 512
LINE_DETECTION_IMG_SIZE = 768


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=None, help="Path to image")
    parser.add_argument("--img-dir", type=str, default=None, help="Path to directory with images")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image: str = args.image
    img_dir: str = args.img_dir

    with open("../config/inference.yaml", 'r') as f:
        config = yaml.safe_load(f)

    pipeline = build_line_word_pipeline(config)
    
    if not img_dir and image:
        pipeline.predict_on_image(
            image,
            config["prediction_dir"]
        )
    elif img_dir:
        for filename in os.listdir(img_dir):
            if not filename.endswith(".jpg"):
                continue
            print(f" Processing {filename} ".center(70, "-"))
            pred_dir = os.path.join(config["prediction_dir"], filename.split(".")[0])
            pipeline.predict_on_image(
                os.path.join(img_dir, filename),
                pred_dir
            )

    print(f"The results are saved in {config['prediction_dir']} folder")

# if __name__ == "__main__":
#     import os
#     import cv2
#     import matplotlib.pyplot as plt
#     from detection.bounding_boxes.plotter import plot_obbs_on_image
#     import detection.bounding_boxes.shapes as sh

#     directory = r"E:\Dyploma\Latina\LatinaProject\datasets\word-in-lines-rotated-black\train"

#     for file in os.listdir(directory):
#         if file.endswith(".jpg"):
#             label_file = file.replace(".jpg", ".txt")
#             image_path = os.path.join(directory, file)
#             label_path = os.path.join(directory, label_file)

#             words = sh.read_shapes(label_path, sh.to_bbox, ["0"])

#             image = cv2.imread(image_path)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             words = list(map(sh.to_obb, words))
#             image = plot_obbs_on_image(image, words)

#             plt.imshow(image)
#             plt.show()

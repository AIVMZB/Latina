import os
import yaml

from model_wrapper import build_line_word_pipeline
import argparse


WORD_DETECTION_IMG_SIZE = 512
LINE_DETECTION_IMG_SIZE = 768


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image = args.image

    with open("../config/inference.yaml", 'r') as f:
        config = yaml.safe_load(f)

    pipeline = build_line_word_pipeline(config)
    
    pipeline.predict_on_image(
        image,
        config["prediction_dir"]
    )

    print(f"The results are saved in {config['prediction_dir']} folder")


# TODO:
# 2) Move plot function to separate module
# 3) Rename shapes_util to shapes
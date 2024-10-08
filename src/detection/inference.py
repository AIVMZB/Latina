import os
import yaml
import argparse

from detection.model_wrapper import build_line_word_pipeline


WORD_DETECTION_IMG_SIZE = 512
LINE_DETECTION_IMG_SIZE = 768


def build_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--img-dir", type=str, default=None, help="Path to directory with images")

    return parser


def run(args: argparse.Namespace):
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

from ultralytics import YOLO
import torch
import os
import yaml

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
print(f"[DEVICE] - {dev}")


TORCH_DEVICE = torch.device(dev)
YOLO_DEVICE = [0] if dev == "cuda:0" else "cpu"   

IMG_SIZE = 1024
YOLO_MOLDEL = os.path.join("models", "yolov8n.pt")
WORD_DETECT_BEST_MODEL = os.path.join("models", "word_detect_best.pt")
LINES_OBB_BEST = os.path.join("models", "lines_obb_best.pt")


def fixup_yaml(yaml_path: str) -> None:
    """Fix the absolute path to datasets in a yaml configuration"""
    if not os.path.exists(yaml_path):
        print(f"ERROR: Can not fixup'{yaml_path}', REASON: Does not exist")
        return

    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    if "path" in data:
        data["path"] = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "../datasets", 
                os.path.splitext(os.path.basename(yaml_path))[0]
            )
        )

    with open(yaml_path, 'w') as file:
        yaml.dump(data, file, indent=2)

def train_detection_model(
        model_path: str = YOLO_MOLDEL, 
        data_file: str = f"{os.path.dirname(os.path.abspath(__file__))}/words_data.yaml",
        epochs: int = 50
    ) -> None:
    """Trains a model. The results of training will be stored at run/detect directory"""

    fixup_yaml(data_file)
    model = YOLO(model_path).to(TORCH_DEVICE)
    model.train(
        data=data_file, epochs=epochs, batch=1,
        imgsz=IMG_SIZE, device=YOLO_DEVICE, workers=1,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.1,
        scale=0.1, fliplr=0.0, mosaic=0.0, erasing=0.0,
        crop_fraction=0.1
    )


def inference(
        images_dir: str,
        model_path: str = WORD_DETECT_BEST_MODEL, 
        min_confidence: float = 0.28,
        show=False
    ) -> None:
    """Inferences a model. The results will be saved at prediction directory"""
    
    model = YOLO(model_path).to(TORCH_DEVICE)

    image_names = os.listdir(images_dir)

    images = list(map(lambda name: os.path.join(images_dir, name), image_names))

    results = model.predict(images, conf=min_confidence)

    if not os.path.exists("../predictions"):
        os.mkdir("../predictions")

    for i, result in enumerate(results):
        result.plot(labels=True, probs=False, show=show, save=True, line_width=2,
                    filename=os.path.join("../predictions", image_names[i]))


if __name__ == "__main__":
    # Uncomment the line below to train to detect lines oriented boxes
    """
    train_detection_model(
        epochs=5,
        data_file=f"{os.path.dirname(os.path.abspath(__file__))}/words_lines_data.yaml",
        model_path="yolov8n-obb.pt"
    )
    """

    # Uncomment the line below to train to detect words bboxes
    """
    train_detection_model(
        epochs=5,
        data_file=f"{os.path.dirname(os.path.abspath(__file__))}/words_data.yaml",
        model_path="yolov8n.pt"
    )
    """

    # Uncomment the line below to test model on detecting words
    #inference("../images", WORD_DETECT_BEST_MODEL, min_confidence=0.3)

    # Uncomment the line below to test model on detecting lines
    # inference("../datasets/lines-obb/valid/images", LINES_OBB_BEST, min_confidence=0.1, show=True)
    
    ...
from ultralytics import YOLO
import torch
import os


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
print(f"[DEVICE] - {dev}")


TORCH_DEVICE = torch.device(dev)
YOLO_DEVICE = [0] if dev == "cuda:0" else "cpu"   

IMG_SIZE = 900
YOLO_MOLDEL = os.path.join("models", "yolov8n.pt")
BEST_TRAINED_MODEL = os.path.join("models", "best.pt")


def train_detection_model(model_path: str = YOLO_MOLDEL, epochs: int = 50) -> None:
    """Trains a model. The results of training will be stored at run/detect directory"""

    model = YOLO(model_path).to(TORCH_DEVICE)
    model.train(data="data.yaml", epochs=epochs, batch=1, imgsz=IMG_SIZE, device=YOLO_DEVICE, workers=2)


def inference(images_dir: str, 
              model_path: str = BEST_TRAINED_MODEL, 
              min_confidence: float = 0.28,
              show=False) -> None:
    """Inferences a model. The results will be saved at prediction directory"""
    
    model = YOLO(model_path).to(TORCH_DEVICE)

    image_names = os.listdir(images_dir)

    images = list(map(lambda name: os.path.join(images_dir, name), image_names))

    results = model(images, conf=min_confidence)

    if not os.path.exists("../predictions"):
        os.mkdir("../predictions")

    for i, result in enumerate(results):
        result.plot(labels=False, probs=False, show=show, save=True, line_width=2,
                    filename=os.path.join("predictions", image_names[i]))


if __name__ == "__main__":
    # train_detection_model(epochs=5)
    inference("../images", BEST_TRAINED_MODEL)
    
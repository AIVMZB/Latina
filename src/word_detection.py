from ultralytics import YOLO
import torch
import os
from bbox_utils.word_to_lines import crop_line_from_image
from bbox_utils.shapes_util import Obb
import cv2


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
print(f"[DEVICE] - {dev}")


TORCH_DEVICE = torch.device(dev)
YOLO_DEVICE = [0] if dev == "cuda:0" else "cpu"   

IMG_SIZE = 1024
YOLO_MODEL = os.path.join("..", "models", "yolov8n.pt")
WORD_DETECT_BEST_MODEL = os.path.join("..", "models", "word_detect_best.pt")
LINES_OBB_BEST = os.path.join("..", "models", "lines_obb_best.pt")


def train_detection_model(model_path: str = YOLO_MODEL, 
                          data_file: str = "yamls/words_data.yaml",
                          epochs: int = 50,
                          imgsz: int = IMG_SIZE) -> None:
    """Trains a model. The results of training will be stored at run/detect directory"""

    model = YOLO(model_path).to(TORCH_DEVICE)
    model.train(data=data_file, epochs=epochs, batch=1, imgsz=imgsz, device=YOLO_DEVICE, workers=1,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.1, scale=0.1, fliplr=0.0, mosaic=0.0, erasing=0.0,
                crop_fraction=0.1)


def inference(images_dir: str,
              model_path: str = WORD_DETECT_BEST_MODEL, 
              min_confidence: float = 0.28,
              show=False) -> None:
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

def predict_by_words_in_lines(image_path: str, 
                              line_model_path: str, 
                              word_model_path: str,
                              min_confidence: float = 0.1) -> None: 
    if not os.path.exists("../predictions"):
        os.mkdir("../predictions")

    line_model = YOLO(line_model_path).to(TORCH_DEVICE)
    word_model = YOLO(word_model_path).to(TORCH_DEVICE)

    line_results = line_model.predict([image_path], conf=min_confidence)
    
    lines = line_results[0].obb.xyxyxyxyn.to("cpu").numpy()
    image = cv2.imread(image_path)
    for i in range(lines.shape[0]):
        line = lines[i]
        line = Obb(line[0, 0], line[0, 1], line[1, 0], line[1, 1], line[2, 0], line[2, 1], line[3, 0], line[3, 1])
        line_image = crop_line_from_image(image, line)

        if line_image.size == 0:
            print("[WARNING] Failed to crop line")
            continue

        word_results = word_model.predict([line_image], conf=0.5)
        
        word_results[0].plot(labels=True, probs=False, show=False, save=True, line_width=2,
                    filename=os.path.join("../predictions", f"{i}.jpg"))


if __name__ == "__main__":
    # Uncomment the line below to train to detect lines oriented bboxes
    # train_detection_model(epochs=5, data_file="yamls/lines_obb_data.yaml", model_path="yolov8n-obb.pt")

    
    # Uncomment the line below to train to detect words bboxes
    # train_detection_model(epochs=150, data_file="yamls/words_data.yaml", model_path="yolov8m.pt")

    # TODO: try to copy files and train for more epochs
    # train_detection_model(epochs=100, data_file="yamls/words_in_lines_data.yaml", model_path="yolov8m.pt", imgsz=700)

    predict_by_words_in_lines("..\images\AUR_1014_VI_21-101 (text).jpg", 
                              LINES_OBB_BEST, 
                              "../runs/detect/train12/weights/best.pt",
                              min_confidence=0.1)
    
    # Uncomment the line below to test model on detecting words
    # inference("../images", LINES_OBB_BEST, min_confidence=0.1)

    # Uncomment the line below to test model on detecting lines
    # inference("../datasets/lines-obb/valid/images", LINES_OBB_BEST, min_confidence=0.1, show=True)
    ...

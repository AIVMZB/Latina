from ultralytics import YOLO
import torch
import os
from bbox_utils.word_to_lines import crop_line_from_image
from bbox_utils.lines_util import extend_line_to_corners
from bbox_utils.shapes_util import Obb
import cv2
from model_wrapper import YoloWrapper
from preprocessing.preprocessor import ImagePreprocessor


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
print(f"[DEVICE] - {dev}")


TORCH_DEVICE = torch.device(dev)
YOLO_DEVICE = [0] if dev == "cuda:0" else "cpu"   

IMG_SIZE = 1024
LINE_DETECTION_IMG_SIZE = 768
YOLO_MODEL = os.path.join("..", "models", "yolov8n.pt")
WORD_DETECT_BEST_MODEL = os.path.join("..", "models", "word_detect_best.pt")
LINES_OBB_BEST = os.path.join("..", "models", "lines_obb_m_best.pt")


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
        line = extend_line_to_corners(line)
        line_image = crop_line_from_image(image, line)

        if line_image.size == 0:
            print("[WARNING] Failed to crop line")
            continue

        word_results = word_model.predict([line_image], conf=0.5)
        
        word_results[0].plot(labels=True, probs=False, show=False, save=True, line_width=2,
                    filename=os.path.join("../predictions", f"{i}.jpg"))


if __name__ == "__main__":
    preprocessor = ImagePreprocessor.load(r"..\preprocessors\gray-untextured.pkl")
    model = YoloWrapper("yolov8l.pt", TORCH_DEVICE, preprocessor)
    model.train(r"..\yamls\five_classes_data.yaml", epochs=300, img_size=IMG_SIZE)

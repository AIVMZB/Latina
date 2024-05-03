from ultralytics import YOLO
import torch 
import os


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

print(f"[DEVICE] - {dev}")
device = torch.device(dev)


def inference():
    model = YOLO("models/best.pt").to(device)

    image_names = os.listdir("../images")

    images = list(map(lambda name: os.path.join("..", "images", name), image_names))

    results = model(images, conf=0.29, show_labels=False, show_conf=False, line_width=5)

    if not os.path.exists("predictions"):
        os.mkdir("predictions")

    image_idx = 0
    for result in results:
        result.plot(labels=False, probs=False, show=False, save=True,
                    filename=os.path.join("predictions", image_names[image_idx]))
        
        image_idx += 1

if __name__ == "__main__":
    inference()

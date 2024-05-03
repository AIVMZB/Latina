from ultralytics import YOLO
import torch

IMG_SIZE = 900

def main():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    print(f"[DEVICE] - {dev}")

    device = torch.device(dev)

    model = YOLO('models/best.pt').to(device)

    results = model.train(data="data.yaml", epochs=50, batch=1, imgsz=IMG_SIZE, device=[0], workers=2)


if __name__ == "__main__":
    main()

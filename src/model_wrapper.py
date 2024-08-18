from preprocessing.preprocessor import ImagePreprocessor
from ultralytics import YOLO
from typing import Union
import numpy as np
import cv2
import torch
import os


# TODO: Write docstrings

class YoloWrapper:
    TRAIN_KWARGS = dict(batch=1, workers=1, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, 
                        translate=0.1, scale=0.1, fliplr=0.0, mosaic=0.0, erasing=0.0, crop_fraction=0.1)

    def __init__(
            self, 
            model_path: str, 
            device: Union[torch.device, str],
            preprocessor: Union[ImagePreprocessor, str]
            ):
        if isinstance(device, torch.device):
            self._device = device
        elif isinstance(device, str):
            self._device = torch.device(device)
        
        self._model = YOLO(model_path).to(self._device)
        
        if isinstance(preprocessor, ImagePreprocessor):
            self._preprocessor = preprocessor
        elif isinstance(preprocessor, str):
            assert os.path.exists(preprocessor), "Provide preprocessor object or its valid path"
            self._preprocessor = ImagePreprocessor.load(preprocessor)


    def train(self, data_file: str, epochs: int, img_size: int) -> None:
        self._model.train(data=data_file, epochs=epochs, imgsz=img_size, device=self._device, **YoloWrapper.TRAIN_KWARGS)


    def inference_image(self, image_path: str, prediction_dir: str, min_conf: float = 0.5, show_plot: bool = True):
        image = cv2.imread(image_path)
        image = self._preprocessor.process(image)

        result = self._model.predict([image], conf=min_conf)[0]
        
        if not os.path.exists(prediction_dir):
            os.mkdir(prediction_dir)
        
        result.plot(labels=True, probs=False, show=show_plot, save=True, line_width=2,
                    filename=os.path.join(prediction_dir, os.path.basename(image_path)))


    @property
    def model(self) -> YOLO:
        return self._model


    @model.setter
    def model(self, weiths: str):
        self._model = YOLO(weiths).to(self._device)
    

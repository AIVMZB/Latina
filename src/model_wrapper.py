from preprocessing.preprocessor import ImagePreprocessor
from bbox_utils.shapes_util import Obb
from bbox_utils.word_to_lines import crop_line_from_image
from bbox_utils.lines_util import extend_line_to_corners

from ultralytics import YOLO
from typing import Union
import cv2
import torch
import os


class YoloWrapper:
    TRAIN_KWARGS = dict(batch=2, workers=1, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, 
                        translate=0.1, scale=0.1, fliplr=0.0, mosaic=0.0, erasing=0.0, crop_fraction=0.1)

    def __init__(
            self, 
            model_path: str, 
            device: Union[torch.device, str],
            preprocessor: Union[ImagePreprocessor, str, None] = None
        ):
        """
        Creates instance of YoloWrapper
        Args:
            model_path (str): path to checkpoint of model weights
            device (torch.device | str): name of torch device or torch device itself
            preprocessor (ImagePreprocessor | str | None): instance of ImagePreprocessor or path to its folder. Optional
        """
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
        else:
            self._preprocessor = ImagePreprocessor()

    def train(self, data_file: str, epochs: int, img_size: int) -> None:
        """
        Trains model using given data
        Args:
            data_file (str): path to yaml file of dataset
            epochs (int): number of epochs to train
            img_size (int): image size
        """
        self._model.train(data=data_file, epochs=epochs, imgsz=img_size, device=self._device, **YoloWrapper.TRAIN_KWARGS)

    def inference_image(self, image_path: str, prediction_dir: str, min_conf: float = 0.5, show_plot: bool = True):
        """
        Inferences model work in image
        Args:
            image_path (str): Path to image
            prediction_dir (str): Path to directory where to save results
            min_conf (float): Minimal confidence of prediction. Defaults to 0.5
            show_plot (bool): If set to True, shows plot of model's prediction
        """
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
    

class LineWordPipeline:
    def __init__(self, line_detection_model: str, word_detection_model: str, device: str = "cuda:0"):
        """
        Creates instance of LineWordPipeline
        Args:
            line_detection_model (str): path to line detection model checkpoint
            word_detection_model (str): path to word detection model checkpoint
            device (str): name of torch device. Defaults to "cuda:0"
        """
        self._device = torch.device(device)
        self._line_model = YOLO(line_detection_model).to(self._device)
        self._word_model = YOLO(word_detection_model).to(self._device)

    def predict_on_image(self, image_path: str, output_path: str, 
                         plot_lines: bool = False,
                         line_conf: float = 0.5, word_conf: float = 0.4):
        """
        Crops detected lines from image and predicts words on it
        Args:
            image_path (str): path to image
            output_path (str): path to save results
            plot_lines (bool): If True, plots predicted lines on image. Defaults to False
            line_conf (float): Minimal confidence of lines prediction. Defaults to 0.5
            word_conf (float): Minimal confidence of words prediction. Defaults to 0.4
        """
        os.makedirs(output_path, exist_ok=True)

        line_results = self._line_model.predict([image_path], conf=line_conf)
        if plot_lines:
            line_results[0].plot(labels=True, probs=False, show=True, save=True, line_width=2, 
                                 filename=os.path.join(output_path, "lines.jpg"))
            
        lines = line_results[0].obb.xyxyxyxyn.to("cpu").numpy()
        image = cv2.imread(image_path)
        for i in range(lines.shape[0]):
            line = lines[i]
            line = Obb(line[0, 0], line[0, 1], line[1, 0], line[1, 1], line[2, 0], line[2, 1], line[3, 0], line[3, 1])
            line = extend_line_to_corners(line)
            line_image = crop_line_from_image(image, line)

            if line_image.size == 0:
                print("[WARNING] Failed to crop line. Its size is zero!")
                continue

            word_results = self._word_model.predict([line_image], conf=word_conf)
            
            word_results[0].plot(labels=True, probs=False, show=False, save=True, line_width=2,
                        filename=os.path.join(output_path, f"{i}.jpg"))
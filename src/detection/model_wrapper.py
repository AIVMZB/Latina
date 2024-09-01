from detection.preprocessing.preprocessor import ImagePreprocessor
from detection.bounding_boxes.shapes import plot_obbs_on_image
from detection.bounding_boxes.word_to_lines import crop_line_from_image
from detection.bounding_boxes.lines import extend_lines_to_corners
from detection.intersect_resolver import IntersectionResolver, build_resolver_by_name, resolve_intersected_objects, tensor_to_boxes

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
    def __init__(self, 
                 line_detection_model: str, 
                 word_detection_model: str,
                 line_conf: float,
                 word_conf: float,
                 line_int_resolver: IntersectionResolver | None,
                 word_int_resolver: IntersectionResolver | None,
                 device: str = "cuda:0"):
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
        
        self._line_int_resolver = line_int_resolver
        self._word_int_resolver = word_int_resolver

        self._line_conf = line_conf
        self._word_conf = word_conf

    def predict_on_image(self, image_path: str, output_path: str):
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

        line_results = self._line_model.predict([image_path], conf=self._line_conf)
            
        lines = line_results[0].obb.xyxyxyxyn
        line_confs = line_results[0].obb.conf
        
        image = cv2.imread(image_path)
        raw_lines_image = plot_obbs_on_image(image.copy(), tensor_to_boxes(lines), (255, 0, 0))
        cv2.imwrite(os.path.join(output_path, "line_predictions.jpg"), raw_lines_image)
        
        if self._line_int_resolver is not None:
            resolved_lines = resolve_intersected_objects(lines, line_confs, 0.2, self._line_int_resolver)
            resolved_lines = extend_lines_to_corners(resolved_lines)
        
            resolved_lines_image = plot_obbs_on_image(image.copy(), resolved_lines, (255, 0, 0))
            cv2.imwrite(os.path.join(output_path, "intersection_resolved_lines.jpg"), resolved_lines_image)
        else:
            resolved_lines = raw_lines_image


        for i in range(len(resolved_lines)):
            line = resolved_lines[i]
            line_image = crop_line_from_image(image, line)

            if line_image.size == 0:
                print("[WARNING] Failed to crop a line. Its size is zero!")
                continue

            word_results = self._word_model.predict([line_image], conf=self._word_conf)
            
            words = word_results[0].boxes.xyxyn
            word_confs = word_results[0].boxes.conf

            prediction_image = plot_obbs_on_image(line_image.copy(), tensor_to_boxes(words), (255, 0, 0))
            cv2.imwrite(os.path.join(output_path, f"{i}.jpg"), prediction_image)

            if self._word_int_resolver is not None:
                resolved_words = resolve_intersected_objects(words, word_confs, 0.1, self._word_int_resolver)

                prediction_image = plot_obbs_on_image(line_image.copy(), resolved_words, (255, 0, 0))
                cv2.imwrite(os.path.join(output_path, f"{i}-r.jpg"), prediction_image)


def build_line_word_pipeline(config: dict[str, any]):
    """
    Builds LineWordPipeline based on config file
    Args:
        config (dict[str, any]) - parsed yaml file
    Returns:
        line_word_pipeline (LineWordPipeline)
    """
    line_int_resolver = build_resolver_by_name(
        config["line_detection"].get("intersection_resolver")
    )
    word_int_resolver = build_resolver_by_name(
        config["word_detection"].get("intersection_resolver")
    )
    
    return LineWordPipeline(
        config["line_detection"]["model_path"],
        config["word_detection"]["model_path"],
        config["line_detection"]["min_conf"],
        config["word_detection"]["min_conf"],
        line_int_resolver,
        word_int_resolver
    )

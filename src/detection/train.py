from detection.model_wrapper import YoloWrapper
from detection.preprocessing.preprocessor import ImagePreprocessor, EdgeDetection


if __name__ == "__main__":
    preprocessor = ImagePreprocessor(methods=[EdgeDetection()])
    model = YoloWrapper("yolov8m-obb.pt", "cuda:0", preprocessor)
    model.train("E:\Labs\year_3\Latina\LatinaProject\yamls\lines_obb_edged_data.yaml", 300, 768)

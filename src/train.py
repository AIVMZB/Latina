from detection.model_wrapper import YoloWrapper
from detection.preprocessing.preprocessor import ImagePreprocessor, EdgeDetection


if __name__ == "__main__":
    model = YoloWrapper("yolov8m-obb.pt", "cuda:0")
    model.train("E:\Dyploma\Latina\LatinaProject\yamls\lines_obb_data.yaml", epochs=330, img_size=600)

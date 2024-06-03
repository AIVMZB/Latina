import numpy as np
import cv2 as cv
import os
import pickle


"""Requires: opencv-python, numpy"""

class ProcessKernel:
    def __init__(self, 
                 contrast: float = 1.0, invert_colors: bool = True,
                 denoise_intensity: int = 5,
                 sharpen_intensity: float = 7,
                 morphological_kernel: np.ones = np.ones((2, 2), np.uint8)) -> None:
        self.contrast = contrast
        self.invert_colors = invert_colors
        self.denoise_intensity = denoise_intensity
        self.sharpen_intensity = sharpen_intensity
        self.morphological_kernel = morphological_kernel

    def process_image(self, img: np.ndarray) -> np.ndarray:
        # Brightness & Contrast adjustment
        img = cv.convertScaleAbs(img, 0, self.contrast)

        # Denoiser
        img = cv.fastNlMeansDenoising(img, None, self.denoise_intensity, 50, 20)

        # Sharpen filter
        kernel = np.array([
            [1,  -1, -1],
            [-1, self.sharpen_intensity, -1],
            [-1, -1, -1]
        ])
        img = cv.filter2D(img, 0, kernel)

        # Not needed most of the time (play with it)
        # Morphological operations to remove noise and fill gaps
        # img = cv.dilate(img, self.morphological_kernel, iterations=1)
        # img = cv.erode(img, self.morphological_kernel, iterations=1)

        # Binarization and invert
        _, img = cv.threshold(
            img, 0, 255,
            cv.THRESH_BINARY_INV + (
                cv.THRESH_MASK if not self.invert_colors else cv.THRESH_OTSU
            )
        )

        return img

    def save(self, path: str) -> str:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        
        return path

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def __repr__(self) -> str:
        return f"Contrast: {self.contrast}, Invert: {self.invert_colors}, Denoise: {self.denoise_intensity}," + \
               f"Sharpen: {self.sharpen_intensity}, Morphological: {self.morphological_kernel}"

class Preprocessor:
    def __init__(
        self,
        input_path: str, output_path: str = None,
        kernel_path: str = None,
        kernel: ProcessKernel = None,
        contrast: float = 1.0, invert_colors: bool = True,
        denoise_intensity: int = 5,
        sharpen_intensity: float = 7,
        morphological_kernel: np.ones = np.ones((2, 2), np.uint8)
    ):
        # Path variables
        self.input_path  = input_path
        self.output_path = output_path if output_path else input_path

        if kernel is not None:
            self.kernel = kernel
        elif kernel_path is None:
            self.kernel = ProcessKernel(
                contrast=contrast, invert_colors=invert_colors,
                denoise_intensity=denoise_intensity,
                sharpen_intensity=sharpen_intensity,
                morphological_kernel=morphological_kernel
            )
        else:
            assert os.path.exists(kernel_path), "Kernel path does not exist"
            self.kernel = ProcessKernel.load(kernel_path)
    
    def process(self, image_file):
        # Load initial image
        img = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        img = np.array(img)

        img = self.kernel.process_image(img)

        # Save the processed image
        output_file = image_file.replace(self.input_path, self.output_path)
        cv.imwrite(output_file, img)

    def execute(self, log: bool = False):
        for file in os.listdir(self.input_path):
            if file.endswith((".jpg", ".jpeg", ".png")):
                self.process(os.path.join(self.input_path, file))
                if log:
                    print(f"{file} was proccessed")


if __name__ == "__main__":
    # kernel = ProcessKernel(invert_colors=False, denoise_intensity=3)

    # preprocessor = Preprocessor(
    #     input_path = "..\\datasets\\words\\train",
    #     output_path = "..\\datasets\\words-preprocessed\\train",
    #     kernel=kernel
    # )

    # preprocessor.execute(log=True)


    # preprocessor = Preprocessor(
    #     input_path = "..\\datasets\\words\\validation",
    #     output_path = "..\\datasets\\words-preprocessed\\validation",
    #     kernel=kernel
    # )

    # preprocessor.execute(log=True)

    # kernel.save("kernel.pkl")
    ...
    

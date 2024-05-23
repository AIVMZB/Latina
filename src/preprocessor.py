import numpy as np
import cv2 as cv
import os


"""Requires: opencv-python, numpy"""
class Preprocessor:
    def __init__(
        self,
        input_path: str, output_path: str = None,
        contrast: float = 1.0, invert_colors: bool = True,
        denoise_intensity: int = 5,
        sharpen_intensity: float = 7,
        morphological_kernel: np.ones = np.ones((2, 2), np.uint8)
    ):
        # Path variables
        self.input_path  = input_path
        self.output_path = output_path if output_path else input_path

        # Image processing settings
        self.contrast = contrast
        self.invert_colors = invert_colors
        self.denoise_intensity = denoise_intensity
        self.sharpen_intensity = sharpen_intensity
        self.morphological_kernel = morphological_kernel
    
    def process(self, image_file):
        # Load initial image
        img = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        img = np.array(img)

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
        #img = cv.dilate(img, self.morphological_kernel, iterations=1)
        #img = cv.erode(img, self.morphological_kernel, iterations=1)

        # Binarization and invert
        _, img = cv.threshold(
            img, 0, 255,
            cv.THRESH_BINARY_INV + (
                cv.THRESH_MASK if not self.invert_colors else cv.THRESH_OTSU
            )
        )

        # Save the processed image
        output_file = image_file.replace(self.input_path, self.output_path)
        cv.imwrite(output_file, img)

    def execute(self):
        for file in os.listdir(self.input_path):
            if file.endswith((".jpg", ".jpeg", ".png")):
                self.process(os.path.join(self.input_path, file))


if __name__ == "__main__":
    preprocessor = Preprocessor(
        input_path = "images/",
        output_path = "processed/"
    )

    preprocessor.execute()

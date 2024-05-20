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
    ):
        # Path variables
        self.input_path  = input_path
        self.output_path = output_path if output_path else input_path

        # Image processing settings
        self.contrast = contrast
        self.invert_colors = invert_colors
        self.denoise_intensity = denoise_intensity
        self.sharpen_intensity = sharpen_intensity
    
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

        """ 
            Warning: Do not use, sh*t
            Morphological operations to remove noise and fill gaps
        """
        #kernel = np.ones((2, 2), np.uint8)
        #img = cv.dilate(img, kernel, iterations=1)
        #img = cv.erode(img, kernel, iterations=1)

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

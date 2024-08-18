from abc import ABC, abstractmethod
import numpy as np
import cv2


def _extract_features(image: np.ndarray) -> np.ndarray:
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    features = np.sqrt(sobelx ** 2 + sobely ** 2)

    return features


class PreprocessStep(ABC):
    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class TextureRemove(PreprocessStep):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image - _extract_features(image)


class GrayScale(PreprocessStep):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
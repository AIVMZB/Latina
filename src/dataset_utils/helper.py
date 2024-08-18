import shutil
import os


def copy_images_to_dataset(dataset_path: str, images_path: str):
    base_subdirs = [subdir for subdir in os.scandir(dataset_path) if subdir.is_dir()]

    for subdir in base_subdirs:
        copy_images_to_dir(subdir.path, images_path)


def copy_images_to_dir(dir_path: str, images_path: str):
    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            image_path = os.path.join(images_path, filename.split(".")[0] + ".jpg")
            shutil.copy(image_path, dir_path)


if __name__ == "__main__":
    copy_images_to_dataset(
        r"E:\Labs\year_3\Latina\LatinaProject\datasets\seven-classes",
        r"E:\Labs\year_3\Latina\LatinaProject\images"
    )
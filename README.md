# Medieval documents reader

Link for data - https://drive.google.com/drive/folders/1rDiNWzTFDnUGerJdcM-tIONm1AEr_n0h?usp=sharing

## How to run a model

1) Install CUDA on your PC https://developer.nvidia.com/cuda-11-8-0-download-archive
2) Create venv `py -m venv venv` and activate `venv\Scripts\activate`
3) Run `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` from the https://pytorch.org/get-started/locally/
3) Run `pip install ultralytics`
4) Download and unzip dataset - [text](https://drive.google.com/file/d/1HRDSL47XtTGOfdtr-daI2TK4Bfnhv37i/view?usp=sharing)
5) Download and unzip images to inference - [text](https://drive.google.com/file/d/1UGyXYSRn3QCU3VYMJhO8vPzADc_7tcAY/view?usp=sharing)
6) To run model use `py src/word_detection.py`

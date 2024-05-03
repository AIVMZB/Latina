# Medieval documents reader

Link for data - https://drive.google.com/drive/folders/1rDiNWzTFDnUGerJdcM-tIONm1AEr_n0h?usp=sharing

## How to run a model

1) Install CUDA on your PC https://developer.nvidia.com/cuda-11-8-0-download-archive
2) Create venv `py -m venv venv` and activate `venv\Scripts\activate`
3) Run `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` from the https://pytorch.org/get-started/locally/
3) Run `pip install ultralytics`
4) To inference a model run `py test.py` specifying in the code image folder, saving folder etc.
5) To train a model `py main.py` specifying needed number of epoch, batch size etc. Run results will be stored at `runs\detection\train<n>` directory

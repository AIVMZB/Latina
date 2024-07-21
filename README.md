# Medieval documents reader

The project is created for reading medieval latin documents using deep learning techniques. 

### Set up

1) Clone the repository `git clone https://github.com/AIVMZB/Latina.git`
2) Move into project directory `cd Latina`
3) Install CUDA on your device ([11.8.0](https://developer.nvidia.com/cuda-11-8-0-download-archive) is recomended). 
4) Create virtual enviroment:
    - Windows `py -m venv venv` `venv\Scripts\activate`
    - Linux `python3 -m venv venv` `source venv/bin/activate`
5) Install pytorch for your CUDA version. You can find the command [here](https://pytorch.org/get-started/locally/) 
or run `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` for 11.8.0 version of CUDA. 
6) Install ultralytics `pip install ultralytics`
7) Install other libraries `pip install -r requirements.txt`
8) Download the [datasets](https://drive.google.com/file/d/1Uw4uyqgTVrOy2VdOgZIdBIwMiTCDGNMj/view?usp=drive_link) 
and unpack it to project root folder


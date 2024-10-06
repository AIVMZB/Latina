# Medieval documents reader

The project is created for reading medieval latin documents using deep learning techniques. 

### Set up

1) Clone the repository 
    ```bash
    git clone https://github.com/AIVMZB/Latina.git
    ```
2) Move into project directory 
    ```bash
    cd Latina
    ```
3) Install CUDA on your device ([11.8.0](https://developer.nvidia.com/cuda-11-8-0-download-archive) is recomended). 
4) Create virtual enviroment:
    - Windows 
        ```bash
        py -m venv venv
        venv\Scripts\activate
        ```
    - Linux 
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
5) Install pytorch for your CUDA version. You can find the command [here](https://pytorch.org/get-started/locally/) or for 11.8.0 version of CUDA run 
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

6) Install ultralytics 
    ```bash
    pip install ultralytics
    ```
7) Install other libraries 
    ```bash
    pip install -r requirements.txt
    ```
8) Download the [datasets](https://drive.google.com/file/d/1Uw4uyqgTVrOy2VdOgZIdBIwMiTCDGNMj/view?usp=drive_link) 
and unpack it to project root folder

[Link](https://drive.google.com/drive/folders/1rDiNWzTFDnUGerJdcM-tIONm1AEr_n0h?usp=sharing) for raw data.

[Link](https://drive.google.com/drive/folders/1Xpi9s0vb1pOYyyMBkaao46YcdjZVHw7s?usp=drive_link) for dataset splited by documents.

[Link](https://drive.google.com/drive/folders/1adG6ER5cRvbnMYlqTchT-Awiddgx0Zly?usp=drive_link) for dataset sorted by classes.
# Smart Authentication

Smart KYC server which takes Aadhaar Card images as the input and processes the data for the individual.

It uses some open source project to work properly :

* [Tesseract](https://github.com/tesseract-ocr/tesseract) - Tesseract Open Source OCR Engine
* [Text-Detection-CTPN](https://github.com/eragonruan/text-detection-ctpn/tree/master) - Text detection mainly based on ctpn model in tensorflow


Note : This Project is tested and developed on Linux (Linux Mint) and Python 3.6


---


## Installation

### Install Tesseract Binary for ocr

- For Ubuntu and Ubuntu based Distros

```bash
sudo apt-get install tesseract-ocr
```

For other other OS users refer this link : [Installing Tesseract for OCR](https://www.pyimagesearch.com/2017/07/03/installing-tesseract-for-ocr/)


### Install Python Dependencies

- Create a virtual environment with ```Conda```

```bash
 conda env create --file environment.yml
```
Note : The name of environment in the ```environment.yml``` file is ```ml-old```.

- Python Libraries required (Python 3.6)

```
easydict==1.9
tensorflow==1.10.0
numpy==1.16.4
Flask==1.1.2
Werkzeug==1.0.1
six==1.15.0
Keras==2.0.8
matplotlib==3.2.2
scipy==1.2.1
waitress==1.4.3
Cython==0.29.21
pytesseract==0.3.5
ipython==7.17.0
Pillow==7.2.0
PyYAML==5.3.1
scikit_learn==0.23.2
```

### Files Description and Resources

Details About Files In This Directory

```
main.py :  File for starting the server
└───routes
│   └───/image/upload : processing the card image (ocr + face embeddings)
│   │
│   └───/live : Detect and Recognise face live (if any embeddings saved)
│   |   
│   └───/ : home path, renders index page
```


```utils.py``` : utils for MTCNN based face detection on Card

```processing.py``` : utils for Aadhaar OCR

```requirements.txt``` : python Dependencies for this Project

```environment.yml``` : yml file for the conda environment

#### Resources

Data and model files required to run this Project

```data/ctpn.pb``` : [ctpn.pb file](https://www.dropbox.com/s/k2ihpj2w5alhrqx/ctpn.pb?dl=0)

```model/20170512-110547/``` : [Model folder](https://www.dropbox.com/sh/lofms9orae4sgz4/AABHfH-rptUj-IZcQEZvGCaBa?dl=0)

Download above two files and put them in the file struture as given below :

```
AuthX
└───data
│   └───ctpn.pb : ctpn.pb file
|
└───model
|   └───20170512-110547
|   |   └───20170512-110547.pb
|   |   └───model-20170512-110547.ckpt-250000.data-00000-of-00001
|   |   └───model-20170512-110547.ckpt-250000.index
|   |   └───model-20170512-110547.meta
```


## Running the project

For Running the project, go to the project Directory and run the command below :

```bash
python main.py
```
this should start a server at port 5000



## Tasks To Do


- [ ] Live video feed extraction
- [ ] In browser video feed
- [ ] Improve CSS/HTML
- [ ] Refactor Code

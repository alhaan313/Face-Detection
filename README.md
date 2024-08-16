# Face Detection and Recognition

This project demonstrates how to perform face detection and recognition using OpenCV's Local Binary Patterns Histograms (LBPH) face recognizer. The system is capable of detecting faces in a live video stream and recognizing them based on a trained model.

## Table of Contents

- [Installation](#installation)
  - [Virtual Environment](#virtual-environment)
  - [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Running the Project](#running-the-project)
- [Usage](#usage)
- [Credits](#credits)

## Installation

### Virtual Environment

To avoid dependency conflicts, it is recommended to create a Python virtual environment.

1. **Navigate to your project directory:**
   ```bash
   cd "Project Directory"

2. **Create a virtual environment:**
   ```bash
   python -m venv FaceDetection

3. **Activate the virtual environment:**
- On Windows
   ```bash
   FaceDetection\Scripts\activate 
- On Mac/linux
  ```bash
  source FaceDetection/bin/activate
### Dependencies
Once the virtual environment is activated, install the required Python packages using requirements.txt.
1. Install the dependencies using requirements.txt
   ```bash
   pip install -r requirements.txt

## Dataset
The project uses a dataset of face images for training. The dataset should be organized in a folder named Dataset, where each subfolder corresponds to a person, and each subfolder contains images of that person.
```bash
Dataset/
├── Person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Person2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```
You can add new images to the dataset by creating a new subfolder within the Dataset directory and placing the corresponding images in it.

## Running the Project
Once everything is set up, you can run the face detection and recognition script.
1. Run the script:
   ```bash
   python main.py
   ```
2. Live Video Stream:
   
   The script will start the webcam and detect faces in the live video stream. If a face is recognized, the person's name and confidence score will be displayed on the screen.

4. Exit:
   
   To exit the video stream, press the q key.

## Usage
- Adding New Faces:
  - To add a new person to the recognition system, create a new folder in the Dataset directory with the person's name and add their face images to this folder. Then, retrain the model by rerunning the script.
- Confidence Score:
  - The confidence score indicates the reliability of the recognition. A lower score means a more confident match.

## Credits
This project uses the following technologies:
- [OpenCV](https://opencv.org/) - For computer vision tasks, including face detection and recognition.
- [NumPy](https://numpy.org/) - For numerical operations in Python.


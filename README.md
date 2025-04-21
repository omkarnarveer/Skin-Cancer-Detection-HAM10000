# Skin Cancer Detection using HAM10000 Dataset

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning-based web application for detecting skin cancer lesions using the HAM10000 dataset. 

The model achieves high accuracy in classifying 7 different types of skin lesions.

## Features

- Web-based interface for image upload

- Deep learning model with 90%+ accuracy

- Real-time prediction results

- Responsive UI with Bootstrap

- Confidence percentage display

- Support for multiple image formats

## Installation

1. Clone the repository:

git clone https://github.com/omkarnarveer/Skin-Cancer-Detection-HAM10000.git

cd Skin-Cancer-Detection-HAM10000

2. Create and activate virtual environment:

python -m venv venv

source venv/bin/activate

3. Install dependencies:

pip install -r requirements.txt

4. Create necessary directories:

mkdir -p static/uploads static/models data/raw data/processed

# Usage

1. Run the Flask application:

python train.py

python app.py

#Training Details

Model Architecture

Input: 224x224 RGB images

4 Convolutional blocks with BatchNorm and MaxPooling

Final Dense layers with Dropout

Output: 7 classes with Softmax activation

Hyperparameters
Image Size: 224x224

Batch Size: 32

Epochs: 50

Learning Rate: 0.0001

Optimizer: Adam

Loss Function: Categorical Crossentropy

Data Augmentation
Rotation: ±20 degrees

Width/Height Shift: 10%

Shear: 10%

Zoom: 10%

Horizontal Flip: True

API Endpoints

GET /: Home page with upload form

POST /predict: Prediction endpoint for image uploads

Requirements

See requirements.txt

Contributing

Fork the repository

Create your feature branch (git checkout -b feature/fooBar)

Commit your changes (git commit -am 'Add some fooBar')

Push to the branch (git push origin feature/fooBar)

Create a new Pull Request

License
Distributed under the MIT License. See LICENSE for more information.

Acknowledgments
HAM10000 dataset authors

TensorFlow/Keras team

Flask documentation


This README includes:
1. Project overview and features
2. Clear installation instructions
3. Dataset preparation guide
4. Training details and model architecture
5. API documentation
6. Contribution guidelines
7. License information

The requirements.txt file includes all necessary dependencies with versions that are known to work together. Users can install them with a single command as shown in the installation section
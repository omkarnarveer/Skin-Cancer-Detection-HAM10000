# 🩺 Skin Cancer Detection using HAM10000 Dataset

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/Skin-Cancer-Detection-HAM10000/)

A deep learning-powered web application for early detection of skin cancer lesions using convolutional neural networks (CNNs) trained on the HAM10000 dataset.

![Web Interface Demo](static/demo.gif)

## ✨ Features

- 🖥️ Web-based interface for easy image uploads
- 🧠 Deep learning model with >90% validation accuracy
- 📊 Real-time prediction results with confidence percentages
- 📱 Responsive mobile-friendly design
- 📈 Interactive prediction probability visualization
- ⚕️ Supports 7 types of skin lesions detection
- 🛡️ File validation and size restrictions (5MB max)

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- HAM10000 dataset (available [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T))

### Setup
# Clone repository
git clone https://github.com/omkarnarveer/Skin-Cancer-Detection-HAM10000.git

cd Skin-Cancer-Detection-HAM10000

# Create virtual environment
python -m venv venv

source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directory structure
mkdir -p static/uploads static/models dataset/raw

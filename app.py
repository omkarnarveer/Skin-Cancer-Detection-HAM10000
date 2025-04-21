from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import cv2
import os

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'static/models/skin_cancer_model.h5'
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = {
    0: 'Actinic Keratoses',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanocytic Nevi',
    5: 'Melanoma',
    6: 'Vascular Lesions'
}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  # Match model's expected input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html', CLASS_NAMES=CLASS_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            processed_img = preprocess_image(filepath)
            preds = model.predict(processed_img, verbose=0)
            class_idx = np.argmax(preds)
            confidence = np.max(preds) * 100

            result = {
                'class': CLASS_NAMES[class_idx],
                'confidence': round(confidence, 2),
                'filename': filename,
                'all_predictions': {
                    cls_name: f"{preds[0][idx] * 100:.2f}%"
                    for idx, cls_name in CLASS_NAMES.items()
                }
            }
            return render_template('result.html', result=result)
        except Exception as e:
            print(f"Error processing image: {e}")
            return redirect(url_for('home'))

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
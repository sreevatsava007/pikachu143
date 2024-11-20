import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2

app = Flask(__name__)

# Load the model once when the app starts (avoiding reloading on every request)
model = tf.keras.models.load_model('improved_plastic_classifier.h5')

# Set up upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Prediction function
def predict_image(img_path):
    # Load and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Mapping classes to labels (assuming 0: non-plastic, 1: plastic)
    class_labels = {0: 'Non-Plastic', 1: 'Plastic'}

    return class_labels.get(predicted_class, 'Unknown')

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get the prediction for the uploaded image
        prediction = predict_image(file_path)

        return jsonify({'prediction': prediction})

    return jsonify({'error': 'Invalid file type. Only .jpg, .jpeg, and .png are allowed.'}), 400

if __name__ == '__main__':
    # Run with Gunicorn for better production performance
    app.run(debug=True, threaded=True)

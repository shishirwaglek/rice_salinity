from flask import Flask, request, jsonify, redirect, url_for, render_template 
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
import joblib
import os
from dotenv import load_dotenv
import tempfile
from azure.storage.blob import BlobServiceClient
from io import BytesIO

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv('API_KEY')
AZURE_CONNECTION_STRING = os.getenv('AZURE_CONNECTION_STRING')

# Initialize Flask app
app = Flask(__name__, 
             template_folder='../frontend/templates', 
             static_folder='../frontend/static')  
CORS(app)  # Enable CORS to allow cross-origin requests

@app.route('/get-api-key')
def get_api_key():
    return jsonify({"api_key": API_KEY})

@app.route('/')
def index1():
    return render_template('index.html')

@app.route('/about-prj')
def index2():
    return render_template('about-prj.html')

@app.route('/about-us')
def index3():
    return render_template('about-us.html')

@app.route('/result')
def index4():
    return render_template('result.html')

@app.route('/fail')
def index5():
    return render_template('fail.html')

# Azure Blob Storage settings
CONTAINER_NAME = "ml-models"  

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

# Model filenames mapping
models_info = {
    "model_1": "VGG16_model.h5",
    "model_2": "inceptionv3SDG.h5",
    "model_3": "ResNet50_model.h5",
    "model_4": "EfficientNetB3_model.h5",
    "plant_detection_model": "plant_non_plant_model.h5",
    "meta_learner": "meta_learner.pkl"
}

def load_model_from_blob(model_name):
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=model_name)

    # Stream the model into an in-memory buffer
    model_stream = BytesIO()
    blob_client.download_blob().readinto(model_stream)
    model_stream.seek(0)  # Reset the pointer to the beginning of the stream

    # Create a temporary file to save the model
    temp_file_path = tempfile.mktemp(suffix='.h5')
    try:
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(model_stream.getvalue())
        
        # Load the model from the temporary file
        model = tf.keras.models.load_model(temp_file_path, compile=False)
    finally:
        os.remove(temp_file_path)  # Clean up the temporary file

    return model

def load_joblib_from_blob(model_name):
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=model_name)

    # Stream the model into an in-memory buffer
    model_stream = BytesIO()
    blob_client.download_blob().readinto(model_stream)
    model_stream.seek(0)  # Reset the pointer to the beginning of the stream

    # Load the joblib model from the in-memory stream
    model = joblib.load(model_stream)
    return model

# Load your pre-trained models
model_1 = load_model_from_blob(models_info["model_1"])          
model_2 = load_model_from_blob(models_info["model_2"])          
model_3 = load_model_from_blob(models_info["model_3"])          
model_4 = load_model_from_blob(models_info["model_4"])          
plant_detection_model = load_model_from_blob(models_info["plant_detection_model"])
meta_learner =   load_model_from_blob(models_info["meta_learner"])  

# Class labels
class_labels = ['Highly_Tolerant', 'Tolerant', 'Moderately_Tolerant', 'Sensitive', 'Highly_Sensitive']

# Preprocess the image for each model
def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)  
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  
    image_array = image_array / 255.0  
    return image_array

def get_predictions(model, processed_image):
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions[0])  
    predicted_label = class_labels[predicted_index]  
    scores = {class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))}
    return predicted_label, scores, predictions[0], predicted_index

def is_rice_plant(image):
    processed_image = preprocess_image(image, (150, 150))  
    prediction = plant_detection_model.predict(processed_image)
    return prediction[0][0] >= 0.5  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]  
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        if not is_rice_plant(image):
            return redirect(url_for('fail'))  

        target_sizes = {
            'model_1': (299, 299),  
            'model_2': (244, 244),  
            'model_3': (244, 244),  
            'model_4': (300, 300),  
        }

        meta_features = []
        predictions = {}
        for i, (model, target_size_key) in enumerate(zip([model_1, model_2, model_3, model_4], target_sizes.keys()), start=1):
            target_size = target_sizes[target_size_key]
            processed_image = preprocess_image(image, target_size)
            predicted_label, scores, prob_distribution, predicted_index = get_predictions(model, processed_image)
            predictions[f'model_{i}'] = {
                'predicted_class': predicted_label,
                'scores': scores
            }
            meta_features.append(predicted_index)

            plt.figure(figsize=(10, 6))
            plt.bar(class_labels, prob_distribution, color='skyblue')
            plt.xlabel('Class Labels')
            plt.ylabel('Probability')
            plt.title(f'Model {i} ({type(model).__name__}) Class Probability Distribution')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)

            img_bytes = io.BytesIO()
            plt.tight_layout()
            plt.savefig(img_bytes, format='png')
            plt.close()
            img_bytes.seek(0)

            graph_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
            predictions[f'model_{i}']['graph'] = f"data:image/png;base64,{graph_base64}"

        meta_features = np.array(meta_features).reshape(1, -1)
        meta_prediction = meta_learner.predict(meta_features)
        final_class = class_labels[meta_prediction[0]]

        try:
            meta_probabilities = meta_learner.predict_proba(meta_features)[0]
            predictions['meta_learner'] = {
                'predicted_class': final_class,
                'scores': {class_labels[i]: float(meta_probabilities[i]) for i in range(len(class_labels))}
            }
        except AttributeError:
            predictions['meta_learner'] = {'predicted_class': final_class}

        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

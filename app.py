from flask import Flask, request, jsonify, redirect, url_for, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io
import matplotlib
import joblib
import os
from dotenv import load_dotenv
import tempfile
from azure.storage.blob import BlobServiceClient
from io import BytesIO

matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv('API_KEY')
AZURE_CONNECTION_STRING = os.getenv('AZURE_CONNECTION_STRING')

# Initialize Flask app
app = Flask(__name__,
             template_folder='./frontend/templates',
             static_folder='./frontend/static')  
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
CONTAINER_NAME = "mlmodels"

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

# Model filenames mapping
models_info = {
    "model_1": "VGG16_model.h5",
    "model_2": "inceptionv3SDG.h5",
    "model_3": "ResNet50_model.h5",
    "model_4": "EfficientNetB3_model.h5",
    "plant_detection_model": "plant_non_plant_model.h5",
 
}

# Local storage path
LOCAL_MODEL_DIR = "/home/site/wwwroot/models/"

# Ensure the directory exists
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

def download_model_if_not_exists(model_name, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {model_name} from Azure Blob Storage...")
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=model_name)
        model_stream = BytesIO()
        blob_client.download_blob().readinto(model_stream)
        model_stream.seek(0)
        with open(local_path, 'wb') as local_file:
            local_file.write(model_stream.getvalue())
        print(f"{model_name} downloaded and saved to {local_path}.")
    else:
        print(f"{model_name} already exists at {local_path}.")

def load_model(local_path):
    return tf.keras.models.load_model(local_path, compile=False)

def load_joblib_model(local_path):
    with open(local_path, 'rb') as file:
        return joblib.load(file)

# Download and load models
for model_key, model_name in models_info.items():
    local_model_path = os.path.join(LOCAL_MODEL_DIR, model_name)
    download_model_if_not_exists(model_name, local_model_path)

# Load TensorFlow models
model_1 = load_model(os.path.join(LOCAL_MODEL_DIR, models_info["model_1"]))
model_2 = load_model(os.path.join(LOCAL_MODEL_DIR, models_info["model_2"]))
model_3 = load_model(os.path.join(LOCAL_MODEL_DIR, models_info["model_3"]))
model_4 = load_model(os.path.join(LOCAL_MODEL_DIR, models_info["model_4"]))

# Load joblib models
plant_detection_model = load_model(os.path.join(LOCAL_MODEL_DIR, models_info["plant_detection_model"]))
meta_learner = joblib.load('./backend/models/meta_learner.pkl')

# Class labels
class_labels = ['Highly_Tolerant', 'Tolerant', 'Moderately_Tolerant', 'Sensitive', 'Highly_Sensitive']
# Preprocess the image for each model
def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)  # Resize to the target size of the specific model
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize the image to [0, 1] as done in training
    return image_array

# Function to get predictions for a specific model
def get_predictions(model, processed_image):
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions[0])  # Get the index with the highest probability
    predicted_label = class_labels[predicted_index]  # Get the label for the predicted class
    scores = {class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))}
    return predicted_label, scores, predictions[0], predicted_index

# Function to check if the uploaded image is a rice plant
def is_rice_plant(image):
    processed_image = preprocess_image(image, (150, 150))  # Adjust target size as per your plant detection model
    prediction = plant_detection_model.predict(processed_image)
    return prediction[0][0] >= 0.5  # Check if the prediction is greater than or equal to 0.5

# @app.route('/get-api-key', methods=['GET'])
# def get_api_key():
#     return jsonify({'api_key': API_KEY})  # Send the API key securely as a response

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Decode the base64 image from the frontend
        image_data = data['image'].split(',')[1]  # Remove the "data:image/png;base64," prefix
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        # Check if the image is a rice plant
        if not is_rice_plant(image):
            return redirect(url_for('fail'))  # Redirect to fail.html if it's not a plant

        # Define the required target sizes for each model
        target_sizes = {
            'model_1': (299, 299),  # VGG16 input size (commonly 224x224)
            'model_2': (244, 244),  # InceptionV3 input size
            'model_3': (244, 244),  # ResNet50 input size
            'model_4': (300, 300),  # EfficientNetB3 input size
        }

        # Meta features array to collect predictions from all base models
        meta_features = []

        # Make predictions using all models
        predictions = {}
        for i, (model, target_size) in enumerate(zip([model_1, model_2, model_3, model_4], target_sizes.values()), start=1):
            # Preprocess the image according to the required size for the current model
            processed_image = preprocess_image(image, target_size)

            # Get the prediction results for the model
            predicted_label, scores, prob_distribution, predicted_index = get_predictions(model, processed_image)
            predictions[f'model_{i}'] = {
                'predicted_class': predicted_label,
                'scores': scores
            }

            # Collect meta-features for the meta-learner
            meta_features.append(predicted_index)

            # Plot the prediction graph for each model
            plt.figure(figsize=(10, 6))
            plt.bar(class_labels, prob_distribution, color='skyblue')
            plt.xlabel('Class Labels')
            plt.ylabel('Probability')
            plt.title(f'Model {i} ({type(model).__name__}) Class Probability Distribution')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)

            # Save the plot to a BytesIO object instead of a file
            img_bytes = io.BytesIO()
            plt.tight_layout()
            plt.savefig(img_bytes, format='png')
            plt.close()  # Close the plot to free memory
            img_bytes.seek(0)

            # Encode the graph to base64
            graph_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
            predictions[f'model_{i}']['graph'] = f"data:image/png;base64,{graph_base64}"

        # Convert the meta-features to a numpy array and reshape for the meta-learner
        meta_features = np.array(meta_features).reshape(1, -1)

        # Use the meta-learner to make the final prediction
        meta_prediction = meta_learner.predict(meta_features)
        final_class = class_labels[meta_prediction[0]]

        # Attempt to get probability distribution from the meta-learner
        if hasattr(meta_learner, 'predict_proba'):
            meta_prediction_proba = meta_learner.predict_proba(meta_features)[0]
            meta_scores = {class_labels[i]: float(meta_prediction_proba[i]) for i in range(len(class_labels))}
        else:
            meta_prediction_proba = [0.0] * len(class_labels)
            meta_prediction_proba[meta_prediction[0]] = 1.0
            meta_scores = {class_labels[i]: float(meta_prediction_proba[i]) for i in range(len(class_labels))}

        # Plot the prediction graph for the meta-learner
        plt.figure(figsize=(10, 6))
        plt.bar(class_labels, meta_prediction_proba, color='magenta')
        plt.xlabel('Class Labels')
        plt.ylabel('Probability')
        plt.title('Meta-Learner Class Probability Distribution')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)

        # Save the plot to a BytesIO object instead of a file
        meta_img_bytes = io.BytesIO()
        plt.tight_layout()
        plt.savefig(meta_img_bytes, format='png')
        plt.close()  # Close the plot to free memory
        meta_img_bytes.seek(0)

        # Encode the meta learner graph to base64
        meta_graph_base64 = base64.b64encode(meta_img_bytes.read()).decode('utf-8')

        # Add the meta-learner's prediction and graph to the response
        predictions['meta_learner'] = {
            'final_class': final_class,
            'scores': meta_scores,
            'graph': f"data:image/png;base64,{meta_graph_base64}"
        }

        return jsonify(predictions)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Failed to make prediction', 'details': str(e)}), 500

@app.route('/fail', methods=['GET'])
def fail():
    return render_template('/frontend/templates/fail.html')  # Render the fail.html template

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=True)
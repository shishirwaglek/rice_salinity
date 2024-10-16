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

# Load environment variables from .env file
load_dotenv()
# Access the API key from the environment variable
API_KEY = os.getenv('API_KEY')  # Make sure 'API_KEY' is the variable name in your .env file

# Initialize Flask app
app = Flask(__name__, 
             template_folder='../frontend/templates', 
             static_folder='../frontend/static')  # Set template and static folders
CORS(app)  # Enable CORS to allow cross-origin requests

@app.route('/get-api-key')
def get_api_key():
    return jsonify({"api_key": os.getenv("API_KEY")})


@app.route('/')
def index1():
    return render_template('index.html')  # Render the index.html template

@app.route('/about-prj')
def index2():
    return render_template('about-prj.html')  # Render the index.html template

@app.route('/about-us')
def index3():
    return render_template('about-us.html')  # Render the index.html template

@app.route('/result')
def index4():
    return render_template('result.html')  # Render the index.html template

@app.route('/fail')
def index5():
    return render_template('fail.html')  # Render the index.html template


# Load your pre-trained models
model_1 = tf.keras.models.load_model('./models/VGG16_model.h5')          # VGG16
model_2 = tf.keras.models.load_model('./models/inceptionv3SDG.h5')      # InceptionV3
model_3 = tf.keras.models.load_model('./models/ResNet50_model.h5')      # ResNet50
model_4 = tf.keras.models.load_model('./models/EfficientNetB3_model.h5')# EfficientNetB3
# Load the plant detection model
plant_detection_model = tf.keras.models.load_model('./models/plant_non_plant_model.h5')
# Load the meta-learner
meta_learner = joblib.load('./models/meta_learner.pkl')

# Class labels for tolerance models
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
    app.run(debug=True)

# AI-Model-Development-for-motion-detection-object-classification
build a machine learning model in Python and set up Azure ML Studio. The primary objective is to detect door openings, classify objects, and count them accurately.
-----------------
To build a machine learning model in Python for detecting door openings, classifying objects, and counting them accurately, and to set up Azure Machine Learning Studio, we can break this process into several steps.
Key Steps:

    Data Collection and Preprocessing: Prepare labeled data of door openings and object classifications.
    Model Training: Train a machine learning model using computer vision techniques (e.g., Convolutional Neural Networks - CNNs).
    Model Evaluation: Evaluate model performance using metrics such as accuracy, precision, and recall.
    Deployment in Azure ML Studio: Use Azure ML Studio to deploy the model as a web service.
    Testing and Integration: Integrate the deployed model into the existing systems.

Below is the Python code template to get you started with the machine learning pipeline:
Step 1: Install Necessary Libraries

Make sure you have installed the necessary libraries, including Azure ML SDK and libraries for machine learning and computer vision.

pip install azureml-sdk
pip install tensorflow opencv-python
pip install scikit-learn matplotlib

Step 2: Import Libraries

We'll import the necessary Python libraries and modules for the project.

import os
import cv2
import numpy as np
import tensorflow as tf
from azureml.core import Workspace, Dataset, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.run import Run
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

Step 3: Set Up Azure ML Workspace

You'll need to authenticate and set up the Azure ML Workspace for experiment tracking and deployment.

# Load the Azure ML workspace
ws = Workspace.from_config()  # This assumes that you have a config.json file with your workspace details

# Create an experiment
experiment_name = 'object-detection-experiment'
experiment = Experiment(workspace=ws, name=experiment_name)

# Set up compute target for training if not available
compute_name = "aml-compute"
compute_target = None
if compute_name not in ws.compute_targets:
    compute_target = AmlCompute(ws, compute_name)

Step 4: Prepare Dataset

Assuming you have labeled video data or images for door openings and object classification, you will need to prepare the dataset.

# Load the dataset (images, videos, etc.) and perform basic preprocessing
data_directory = '/path/to/data'  # Replace with the actual data path

def load_data(data_directory):
    images = []
    labels = []
    
    for filename in os.listdir(data_directory):
        img_path = os.path.join(data_directory, filename)
        
        # Read and preprocess the image (resize, normalize, etc.)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))  # Resize to match model input dimensions
        img = img / 255.0  # Normalize pixel values to [0, 1]
        
        images.append(img)
        labels.append(int(filename.split('_')[0]))  # Assuming filenames have labels in them
        
    return np.array(images), np.array(labels)

# Load data
X, y = load_data(data_directory)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Step 5: Define and Train the Model

For object detection and classification, we will use a CNN model with multiple layers.

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Assuming 10 classes for object classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

Step 6: Evaluate the Model

Evaluate the model's performance on the test set to ensure it meets the desired performance benchmarks.

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

Step 7: Set Up Azure ML Environment and Deployment

Once the model is trained, you can create an environment and deploy the model to Azure.

# Create a Python environment for the experiment
env = Environment.from_conda_specification(
    name='object-detection-env',
    file_path='environment.yml'  # Include dependencies in this file
)

# Create a script config to run the model in the Azure ML environment
src = ScriptRunConfig(source_directory='.', script='train.py', environment=env, compute_target=compute_target)

# Submit the experiment
run = experiment.submit(src)

# Wait for the run to complete
run.wait_for_completion(show_output=True)

Step 8: Deploy the Model as a Web Service in Azure

Once the model is trained, you can deploy it as a web service on Azure for inference. Here’s an example of how you can set up the deployment:

from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import Model

# Register the trained model in Azure ML
model_path = 'outputs/model.pkl'  # Path to your model
model = Model.register(workspace=ws, model_path=model_path, model_name="object-detection-model")

# Define the inference configuration
inference_config = InferenceConfig(entry_script="score.py", environment=env)

# Define the deployment configuration
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy the model as a web service
service = Model.deploy(workspace=ws, name="object-detection-service", models=[model], 
                       inference_config=inference_config, deployment_config=aci_config)

service.wait_for_deployment(show_output=True)

print(f"Service deployed at: {service.scoring_uri}")

Step 9: Scoring Script (score.py)

Create a scoring script to handle requests for predictions from the web service.

import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from azureml.core.model import Model

def init():
    global model
    # Load the trained model when the service starts
    model_path = Model.get_model_path('object-detection-model')
    model = load_model(model_path)

def run(input_data):
    try:
        # Convert input data to numpy array
        data = np.fromstring(input_data, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224)) / 255.0
        
        # Perform prediction
        prediction = model.predict(np.expand_dims(img, axis=0))
        predicted_class = np.argmax(prediction)
        
        return json.dumps({"prediction": predicted_class})
    except Exception as e:
        return json.dumps({"error": str(e)})

Step 10: Integration and Testing

Finally, integrate the deployed model with your application and test the prediction via HTTP requests. You can use tools like Postman or integrate the REST API in your frontend.

import requests

# Example POST request to the deployed service
url = "<SCORING_URI>/score"
data = open('image.jpg', 'rb').read()

response = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'})
print(response.json())  # Prediction result

Conclusion:

This Python script allows you to build a machine learning model for object classification and door opening detection. It leverages TensorFlow for model training, Azure ML Studio for model deployment, and Azure ML SDK for environment setup, model registration, and deployment as a web service. Ensure you have an Azure subscription and configure your workspace appropriately before running the code.

This setup will help you deploy an AI-driven solution for real-time object detection and classification, enabling automated object counting and door-opening detection.

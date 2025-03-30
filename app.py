import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained model
model = load_model('model_vgg16.h5')

# Define ASL labels (matching training data order)
ASL_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
              'U', 'V', 'W', 'X', 'Y', 'Z']

def preprocess_image(image_bytes):
    try:
        # Convert bytes to image using OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (64, 64))
        
        # Ensure image has correct shape (64, 64, 3)
        if image.shape != (64, 64, 3):
            raise ValueError(f"Invalid image shape {image.shape}, expected (64, 64, 3)")
        
        # Normalize the image
        image = image.astype('float32') / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")


def process_image(image):
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        resized_image = cv2.resize(image_rgb, (64, 64))
        
        # Normalize the image
        normalized_image = resized_image.astype('float32') / 255.0
        
        # Add batch dimension
        processed_input = np.expand_dims(normalized_image, axis=0)
        
        # Make prediction
        predictions = model.predict(processed_input, verbose=0)
        if len(predictions) == 0 or len(predictions[0]) != len(ASL_LABELS):
            raise ValueError("Invalid prediction shape")
            
        predicted_class_index = np.argmax(predictions[0])
        if predicted_class_index < 0 or predicted_class_index >= len(ASL_LABELS):
            raise ValueError("Prediction index out of range")
            
        confidence = float(predictions[0][predicted_class_index] * 100)
        
        # Get the predicted letter
        predicted_letter = ASL_LABELS[predicted_class_index]
        
        prediction = {
            'prediction': predicted_letter,
            'confidence': confidence
        }
        
        # Convert processed image to base64
        _, buffer = cv2.imencode('.jpg', resized_image)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return prediction, processed_image
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'Empty file provided'}), 400
            
        # Read image file
        try:
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'error': 'Invalid image format'}), 400
                
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 400
        
        # Process image and get prediction
        try:
            prediction, processed_image = process_image(image)
            
            return jsonify({
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'preprocessed_image': f'data:image/jpeg;base64,{processed_image}'
            })
            
        except Exception as e:
            return jsonify({'error': f'Error making prediction: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True)

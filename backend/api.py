from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os
from tensorflow.keras.models import model_from_json

app = Flask(__name__)
CORS(app)

# Load the custom emotion detection model
def load_emotion_model():
    json_path = os.path.join(os.path.dirname(__file__), 'emotiondetector.json')
    h5_path = os.path.join(os.path.dirname(__file__), 'emotiondetector.h5')
    
    with open(json_path, "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(h5_path)
    return model

# Load model once at startup
emotion_model = load_emotion_model()

# Emotion labels for the custom model
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Preprocessing function for the custom model
def extract_features(image):
    # Convert to grayscale and resize to 48x48
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.resize(gray, (48, 48))
    
    # Normalize and reshape for model input
    gray = gray.astype('float32') / 255.0
    gray = gray.reshape(1, 48, 48, 1)
    return gray

@app.route('/analyze-emotion', methods=['POST'])
def analyze_emotion():
    data = request.json
    img_data = data.get('image')
    if not img_data:
        return jsonify({'error': 'No image provided'}), 400
    try:
        img_bytes = base64.b64decode(img_data.split(',')[1])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Detect faces using Haar cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]
        else:
            face_img = gray
        
        # Preprocess image for the custom model
        processed_img = extract_features(face_img)
        
        # Get emotion prediction
        prediction = emotion_model.predict(processed_img, verbose=0)
        emotion_probs = prediction[0]
        
        # Convert to emotion dictionary
        emotion_dict = {}
        for i, prob in enumerate(emotion_probs):
            emotion_dict[emotion_labels[i]] = float(prob)
        
        # Normalize probabilities
        total = sum(emotion_dict.values())
        if total > 0:
            emotion_dict = {k: v / total for k, v in emotion_dict.items()}
        
        # For age and gender, we'll use placeholder values since the custom model only does emotion
        # You can integrate separate age/gender models here if needed
        response = {
            'age': 25,  # Placeholder - you can add age detection model
            'gender': 'Unknown',  # Placeholder - you can add gender detection model
            'emotion': emotion_dict
        }
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print the full traceback to the terminal
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 
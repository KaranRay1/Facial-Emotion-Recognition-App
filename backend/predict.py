# predict.py
#%%writefile predict.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import os
from io import BytesIO
import time
import subprocess

# Import constants and preprocessing from utils.py
#from utils import preprocess_image_for_vgg16, EMOTION_LABELS, GENDER_LABELS, AGE_LABELS, IMG_WIDTH, IMG_HEIGHT

# Import emote mirroring functions and EmoteAnimator class
# Ensure _EMOTE_ANIMATIONS is imported directly for access as a global
#from emotemirroring import EmoteAnimator, overlay_transparent_image, _EMOTE_ANIMATIONS


# Define your project's main path where model and cascade are located
MAIN_PROJECT_PATH = '/content/drive/MyDrive/Project/Face Emotion Recoginization /face' # Keep this consistent

# --- 1. Load the Trained Model ---
# This path is the specific path to the saved phase1 model (or final model if trained)
MODEL_PATH = os.path.join(MAIN_PROJECT_PATH, '/content/drive/MyDrive/Project/Face Emotion Recoginization /face/multi_task_vgg16_best_model_phase1.h5') # Assume this is the final model name

model = None
if not os.path.exists(MODEL_PATH):
    print("Error: Model not found at {}. Please run model.py to train and save it first.".format(MODEL_PATH))
else:
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully from {}".format(MODEL_PATH))
    except Exception as e:
        print("Error loading model: {}".format(e))
        model = None # Set to None so prediction loop knows not to proceed

# --- 2. Initialize Face Detector (Haar Cascade for simplicity) ---
face_cascade_path = os.path.join(MAIN_PROJECT_PATH, '/content/drive/MyDrive/Project/Face Emotion Recoginization /face/haarcascade_frontalface_default.xml')

if not os.path.exists(face_cascade_path):
    print("Warning: Haar Cascade XML not found at {}. Attempting to download it...".format(face_cascade_path))
    try:
        subprocess.run([
            "wget",
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascades/haarcascade_frontalface_default.xml",
            "-O",
            face_cascade_path
        ], check=True, capture_output=True)
        print("Haar Cascade downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading Haar Cascade: {e.stderr.decode()}")
        print("Failed to download Haar Cascade. Real-time prediction may not work.")
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        print("Failed to download Haar Cascade. Real-time prediction may not work.")

face_cascade = cv2.CascadeClassifier(face_cascade_path)

# --- 3. Initialize Emote Animator ---
EMOTE_TARGET_SIZE = (80, 80) # Size of the emote animation frames
emote_animator = EmoteAnimator(MAIN_PROJECT_PATH, EMOTION_LABELS, target_size=EMOTE_TARGET_SIZE)
if not _EMOTE_ANIMATIONS: # Check if any animations were loaded (_EMOTE_ANIMATIONS is global from emotemirroring)
    print("Emote mirroring disabled due to no animated emotes loaded.")

# --- Colab-specific Webcam Access Functions ---
# Modified JavaScript to include a capture function within the global scope for eval_js
def video_stream_js_snippet():
  js = Javascript('''
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    var imgElement;
    var labelElement;

    var pendingResolve = null;
    var shutdown = false;

    // Make 'capture' function globally accessible for Python's eval_js
    window.capture = function() {
        return new Promise(resolve => {
            pendingResolve = resolve;
            captureCanvas.getContext('2d').drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
            var url = captureCanvas.toDataURL('image/jpeg', 0.8);
            resolve(url); // Resolve the promise with the data URL
        });
    };

    // Make 'stop' function globally accessible
    window.stopCamera = function() {
      if (stream) {
        stream.getVideoTracks()[0].stop();
        video.remove();
        if (div) {
          div.remove();
        }
        video = null;
        div = null;
        stream = null;
        captureCanvas = null;
        imgElement = null;
        labelElement = null;
      }
      shutdown = true;
      console.log('Camera stream stopped by window.stopCamera.');
    };

    async function start() {
      div = document.createElement('div');
      document.body.appendChild(div);

      labelElement = document.createElement('div');
      div.appendChild(labelElement);

      video = document.createElement('video');
      video.style.display = 'block';
      div.appendChild(video);

      try {
        stream = await navigator.mediaDevices.getUserMedia({video: true});
        video.srcObject = stream;
        await video.play();

        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

        captureCanvas = document.createElement('canvas');
        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;
        imgElement = document.createElement('img');
        div.appendChild(imgElement);

        google.colab.output.registerOutputForDisposal(async () => {
          shutdown = true;
          stream.getVideoTracks()[0].stop();
          video.remove();
          div.remove();
          video = null;
          div = null;
          stream = null;
          captureCanvas = null;
          imgElement = null;
          labelElement = null;
          console.log('Camera stream disposed by Colab runtime.');
        });
        console.log('Camera stream started successfully.');
      } catch (err) {
        console.error('Error starting camera stream: ', err);
        labelElement.innerText = 'Camera access denied or error: ' + err.name;
        alert('Camera access denied or error: ' + err.message + '. Please allow camera access.');
        shutdown = true; // Signal Python to stop
      }
    }

    start();
  ''')
  display(js)

# Python function to get a frame from the JavaScript
def video_frame_read(quality=0.8):
  try:
      # Call the globally exposed JavaScript 'capture' function
      data = eval_js('capture()')
      if data:
          return data.split(',')[1] # Return only the base64 part
      else:
          print("JavaScript capture returned empty data.")
          return None
  except Exception as e:
      print(f"Python eval_js error during capture: {e}. Stream might be closed or JS context invalid.")
      # Try to signal JS to stop to clean up
      try: eval_js('window.stopCamera()')
      except: pass
      return None

# --- Prediction Loop (Adapted for Colab Webcam) ---
def run_prediction_loop():
    if model is None:
        print("Model not available. Skipping prediction loop.")
        return

    print("Starting real-time analysis. Grant webcam permissions when prompted.")
    video_stream_js_snippet() # Start the JavaScript webcam stream in the browser
    time.sleep(4) # Increased initial delay for setup and permission granting

    try:
        frame_count = 0
        max_no_frame_retries = 10 # Allow a few retries for dropped frames
        current_no_frame_retries = 0

        while True:
            # Check for early shutdown signal from JS, or if user stopped manually
            js_shutdown_status = eval_js('shutdown')
            if js_shutdown_status:
                print("JavaScript stream signaled shutdown. Exiting loop.")
                break

            # --- Frame Capture ---
            js_reply = video_frame_read()
            
            if js_reply is None:
                current_no_frame_retries += 1
                print(f"No frame received (retry {current_no_frame_retries}/{max_no_frame_retries}). Waiting slightly...")
                time.sleep(0.1) # Small pause to avoid hammering JS
                if current_no_frame_retries >= max_no_frame_retries:
                    print("Max retries for frame capture exceeded. Exiting loop.")
                    break
                continue # Skip processing for this iteration, try getting frame again
            else:
                current_no_frame_retries = 0 # Reset retries on successful frame

            # --- Frame Processing ---
            try:
                image_bytes = b64decode(js_reply)
                image_stream = BytesIO(image_bytes)
                jpg_as_np = np.frombuffer(image_stream.read(), dtype=np.uint8)
                frame = cv2.imdecode(jpg_as_np, flags=1)

                if frame is None:
                    print("Failed to decode frame from bytes (cv2.imdecode returned None), skipping...")
                    continue # Skip current frame, try next

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    processed_face = preprocess_image_for_vgg16(face_roi)
                    processed_face = np.expand_dims(processed_face, axis=0)

                    # --- Prediction ---
                    try:
                        emotion_pred, gender_pred, age_pred = model.predict(processed_face, verbose=0)
                        emotion_label = EMOTION_LABELS[np.argmax(emotion_pred[0])]
                        gender_label = GENDER_LABELS[np.argmax(gender_pred[0])]
                        age_label = AGE_LABELS[np.argmax(age_pred[0])]

                        # --- Drawing and Emote Mirroring ---
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        font_thickness = 2
                        text_color_emotion = (0, 255, 0)
                        text_color_gender = (0, 255, 255)
                        text_color_age = (0, 165, 255)
                        text_y_start = y - 10
                        if text_y_start < 0:
                            text_y_start = y + h + 20

                        cv2.putText(frame, "Emotion: {}".format(emotion_label), (x, text_y_start),
                                    font, font_scale, text_color_emotion, font_thickness, cv2.LINE_AA)
                        cv2.putText(frame, "Gender: {}".format(gender_label), (x, text_y_start + 25),
                                    font, font_scale, text_color_gender, font_thickness, cv2.LINE_AA)
                        cv2.putText(frame, "Age: {}".format(age_label), (x, text_y_start + 50),
                                    font, font_scale, text_color_age, font_thickness, cv2.LINE_AA)

                        if _EMOTE_ANIMATIONS: # Only attempt if emotes were loaded
                            emote_img_to_overlay = emote_animator.get_current_emote_frame(emotion_label)
                            if emote_img_to_overlay is not None:
                                emote_x_offset = x + w - EMOTE_TARGET_SIZE[0] - 10
                                emote_y_offset = y + 10
                                
                                if emote_x_offset < 0: emote_x_offset = 0
                                if emote_y_offset < 0: emote_y_offset = 0
                                if emote_x_offset + EMOTE_TARGET_SIZE[0] > frame.shape[1]:
                                    emote_x_offset = frame.shape[1] - EMOTE_TARGET_SIZE[0]
                                if emote_y_offset + EMOTE_TARGET_SIZE[1] > frame.shape[0]:
                                    emote_y_offset = frame.shape[0] - EMOTE_TARGET_SIZE[1]

                                frame = overlay_transparent_image(frame, emote_img_to_overlay,
                                                                  x_offset=emote_x_offset, y_offset=emote_y_offset)

                    except Exception as e:
                        print(f"Error during prediction or drawing for a face (inner loop prediction block): {e}")
                        cv2.putText(frame, "Pred Error", (x, y - 10),
                                    font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

                # --- Display Frame ---
                _, jpeg_frame = cv2.imencode('.jpg', frame)
                jpeg_bytes = jpeg_frame.tobytes()
                b64_frame = b64encode(jpeg_bytes).decode('utf-8')
                js_code = "imgElement.src = 'data:image/jpeg;base64,{0}';".format(b64_frame)
                eval_js(js_code)
                
            except Exception as e:
                print(f"Error during main frame processing (outer loop): {e}")
                # This could be a decoding issue, continue to next frame
                continue 

    except KeyboardInterrupt:
        print("Stream stopped by user (Ctrl+C detected in Python).")
    except Exception as e:
        print(f"An unexpected error occurred in prediction loop (main try block): {e}")
    finally:
        # Ensure JS stop function is called reliably
        try:
            eval_js('window.stopCamera()')
            print("JavaScript camera stream explicitly stopped.")
        except Exception as e:
            print(f"Error calling JS stopCamera: {e}")
        print("Application closed.")

if __name__ == '__main__':
    run_prediction_loop()
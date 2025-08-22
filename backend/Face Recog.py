import os
import cv2
import numpy as np
from deepface import DeepFace

dir = "Dataset"
os.makedirs(dir, exist_ok=True)

def create_dataset(name):
    person = os.path.join(dir, name)
    os.makedirs(person, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot Open WebCam")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y+h, x:x+w]
            face_path = os.path.join(person, f"{name}_{count}.jpg")
            cv2.imwrite(face_path, face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow("Capture Face in the Camera", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
                break
    cap.release()
    cv2.destroyAllWindows()

def create_training():
    embedding = {}  
    for i in os.listdir(dir):
        person = os.path.join(dir, i)

        if os.path.isdir(person):
            embedding[i] = []  

            for img_name in os.listdir(person):
                img_path = os.path.join(person, img_name)
                try:
                    face_embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                    embedding[i].append(face_embedding)  
                except Exception as e:
                    print(f"Failed to Train images for {i}: {e}")
    return embedding

def recognize_face(embedding):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w] 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
            try:
                analyse = DeepFace.analyze(face_img, actions=["age", "gender", "emotion"], enforce_detection=False)

                if isinstance(analyse, list):
                    analyse = analyse[0]

                age = analyse['age']
                gender = analyse['gender']
                gender = gender if isinstance(gender, str) else max(gender, key=gender.get)
                emotion = max(analyse['emotion'], key=analyse['emotion'].get)

                face_embedding = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)[0]['embedding']
                match = None
                max_similarity = -1

                for i, person_embeddings in embedding.items(): 
                    for embed in person_embeddings:
                        similarity = np.dot(face_embedding, embed) / (np.linalg.norm(face_embedding) * np.linalg.norm(embed))
                        if similarity > max_similarity:
                            max_similarity = similarity
                            match = i
                
                if max_similarity > 0.7:
                    label = f'{match} ({max_similarity:.2f})' 
                else:
                    label = "Unknown Person"

                display_text = f"{label}, Age: {int(age)}, Gender: {gender}, Emotion: {emotion}"
                cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) 

            except Exception as e:
                print("Face Cannot be recognized:", e)

        cv2.imshow("Recognize Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__=="__main__":
    print("1. Create Face Dataset\n 2. Train Dataset\n 3. Recognize Faces\n")
    choices=input("Enter you choice 1 or 2 or 3 😃: ")
    if(choices=='1'):
        name=input("Enter the name of the person: ")
        create_dataset(name)

    elif(choices=='2'):
        embedding=create_training()
        np.save("embedding.npy",embedding)

    elif choices=='3':
        if(os.path.exists("embedding.npy")):
            embedding=np.load("embedding.npy",allow_pickle=True).item()
            recognize_face(embedding)
        else:
            print("Try training th dataset first")

    else:
        print("Invalid Choice")

        



import cv2  
import os
import time
from threading import Thread
import numpy as np 
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances as L2
import mediapipe as mp
from mtcnn import MTCNN
import scipy.stats as st
import warnings
import json
warnings.filterwarnings("ignore")

class FaceRecognition:
    
    def __init__(self,model,path,mode,thresh=1.5,src=0):

        # class attributes related to the model and the video
        self.src = src
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)
        self.recognition = tf.keras.models.load_model(model)
        self.capture = cv2.VideoCapture(src)
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        _, self.img = self.capture.read()
        self.cTime = 0 
        self.pTime = 0
        self.images = os.listdir(path)
        self.names = list(map(lambda x: x.split('.')[0], self.images))
        self.label = []
        self.mode = mode
        self.thresh = thresh
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

        self.detected_faces = []

        self.embedding_file = 'embeddings.json'

        if os.path.exists(self.embedding_file):
            # Load saved embeddings
            with open(self.embedding_file, 'r') as f:
                self.embeddings = np.array(json.load(f))
        else:
            # Compute embeddings and save to file
            for image in self.images:
                img = self.face_detection(os.path.join(path,image))
                self.detected_faces.append(img)
            self.embeddings = self.recognition.predict(np.array(self.detected_faces))
            with open(self.embedding_file, 'w') as f:
                json.dump(self.embeddings.tolist(), f)


        # Thread for running the Face Recognition Concurrently
        self.t = Thread(target=self.recognizeFaces)
        self.t.daemon = True
        self.t.start()

    def detection(self,img, faces):
        for face in faces:
            x,y,w,h = face['box']
        img = img[y:y+h,x:x+w,:]
        img = img/255.
        img = cv2.resize(img,(160,160))
        return img

    def face_detection(self,image):
        print(image)
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector = MTCNN()
        faces = detector.detect_faces(img)
        img = self.detection(img, faces)
        return img

    # Heavy video processing functionality should be defined here
    def recognizeFaces(self):

        while True:
            img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            try:
                img = img[self.y:self.y+self.h,self.x:self.x+self.w,:]
                img = img/255.
                img = cv2.resize(img,(160,160))
                v_emb = self.recognition.predict(np.expand_dims(img,axis=0))
                distances = L2(self.embeddings, v_emb)
                index = np.argmin(distances,axis=0)
                r_index = index[0] if distances[index] < self.thresh else 'Unknown' 
                try:
                    self.label.append(self.names[r_index])
                except TypeError:
                    self.label.append(r_index)
            except:
                pass
            time.sleep(1/60)
        return

    # Running the read/display of the video on the main thread
    def display(self,box_color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (255, 255, 255)
        thickness = 3

        while True:  

            self.img = cv2.flip(self.capture.read()[1],1)
            label = 'Unknown'
            try:
                label = st.mode(self.label[-self.mode:])[0][0]
                (width, height),b = cv2.getTextSize(label, font, fontScale, thickness)
            except IndexError:
                (width, height),b = cv2.getTextSize(label, font, fontScale, thickness)

            results = self.detector.process(self.img)
            if results.detections:
                for detection in results.detections:
                    self.x = int(detection.location_data.relative_bounding_box.xmin*self.width)
                    self.y = int(detection.location_data.relative_bounding_box.ymin*self.height)
                    self.w = int(detection.location_data.relative_bounding_box.width*self.width)
                    self.h = int(detection.location_data.relative_bounding_box.height*self.height)
                    cv2.rectangle(self.img, (self.x, self.y), (self.x+self.w, self.y+self.h), box_color, 2)  
                    cv2.rectangle(self.img, (self.x, self.y-height), (self.x+width, self.y), box_color, -5)
                    cv2.putText(self.img, label, (self.x,self.y), font, fontScale, color, thickness, cv2.LINE_AA)

            self.cTime = time.time()
            fps = 1 / (self.cTime - self.pTime)
            self.pTime = self.cTime
            cv2.putText(self.img, "FPS: "+str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow('Video', self.img)  
            k = cv2.waitKey(20) & 0xff  
            if k==27:  
                break

        self.capture.release()

if __name__ == '__main__':

    encodings_path = 'encodings.npy'

    if os.path.exists(encodings_path):
        # Load encodings from file
        embeddings = np.load(encodings_path)
    else:
        # Compute encodings
        images_path = 'images'
        model_path = 'models/face_embed.h5'
        mode = 7
        thresh = 2.0
        src = 0
        fr = FaceRecognition(model_path, images_path, mode, thresh, src)
        embeddings = fr.embeddings

        # Save encodings to file
        np.save(encodings_path, embeddings)

        try:
            fr.display(box_color=(0,0,0))
        except Exception as e:
            print(e)
    
    
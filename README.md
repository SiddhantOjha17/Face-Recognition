# Face Recognition System 

This code implements a real-time face recognition system using Mediapipe and MTCNN for face detection, and a CNN model for facial recognition.

Face recognition is an important technology in the field of computer vision, which involves the identification of individuals based on their facial features. It has numerous applications in various domains such as security, surveillance, marketing, and entertainment.

The method above achieves face recognition by using a deep learning model to extract embeddings, which are numerical representations of facial features, from a set of reference images. The embeddings are then compared to the embeddings of the face detected in the video stream, and the closest match is identified as the person in the video. The method uses a face detection algorithm to locate faces in the video stream, and the identified person's name is displayed on the screen in real-time.

The implementation of the face recognition system above involves loading a pre-trained deep learning model for face recognition and a face detection algorithm. It uses a multi-threaded approach, where the face recognition process runs on a separate thread to avoid slowing down the video processing. The system first detects and extracts faces from reference images and saves their embeddings to a file. Then, the video stream is processed to detect faces and match them with the saved embeddings to identify the person in the video. The system continuously updates the identified person's name on the screen as new faces are detected. The system's accuracy and performance can be improved by using larger training datasets and more powerful deep learning models.


# Siamese Networks 
The model used to predict the face embedding is based on the concept of Siamese Networks. Siamese Networks are a type of neural network architecture that are designed to compare two inputs and determine how similar they are. 

![alt text](https://user-images.githubusercontent.com/73122995/231527636-8dc00d66-0662-496c-8513-59f563d3c523.png)


Siamese Networks are often used for tasks such as face recognition, where the network needs to compare two images and determine if they belong to the same person. In this case, the two input images would be passed through the Siamese Network, and the output embeddings would be compared to determine if the two images represent the same face.


# Results



https://user-images.githubusercontent.com/96463139/231529285-1196f153-1908-486e-94dd-08cb76d17a8e.mp4



# Requirements 

```bash
pip install -r requirements.txt
```

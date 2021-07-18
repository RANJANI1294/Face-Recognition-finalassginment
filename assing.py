import face_recognition
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

model_face_detect = mp_face_detection.FaceDetection()
model_selfie = mp_selfie_segmentation.SelfieSegmentation(model_selection = 0)

st.title("My first app")
st.write("This project we are going to deal about Face Detection,Face recognition & Selfie segmentation")

add_sb = st.sidebar.selectbox("Choose the option", ("About", "Face Detection", "Face Recognition", "Selfie segmentation"))

if add_sb == "About":
    st.write("Welcome all to this new project.we are going to learn 3 process through different python libraries.")
    st.write("the libraries are opencv-python, media-pipe, numpy, d-lib, face-recognition")

elif add_sb == "Face Detection":
    image_file_path = st.sidebar.file_uploader("Upload file")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        results = model_face_detect.process(image)
        image = image.copy()
        for landmarks in results.detections:
            mp_face_detection.get_key_point(landmarks, mp_face_detection.FaceKeyPoint.NOSE_TIP)
            mp_drawing.draw_detection(image, landmarks)
        st.image(image)

elif add_sb == "Face Recognition":

    image_file_path1 = st.sidebar.file_uploader("image1")
    image_file_path2 = st.sidebar.file_uploader("image2")
    image1 = np.array(Image.open(image_file_path1))

    image2 = np.array(Image.open(image_file_path2))

    st.sidebar.image(image1)
    st.sidebar.image(image2)
    #known_image = face_recognition.load_image_file(np.array(Image.open(image_file_path1)))
    #unknown_image = face_recognition.load_image_file(np.array(Image.open(image_file_path2)))
    #st.sidebar.image(image1)
    #st.sidebar.image(image2)
    biden_encoding = face_recognition.face_encodings(image1)[0]
    unknown_encoding = face_recognition.face_encodings(image2)[0]
    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)

    if results[0]:
        st.write("True!Both the images are identical")
    else:
        st.write("False! Both the images are not identical")
    st.image(image1)
    st.image(image2)

elif add_sb == "Selfie segmentation":
    #MASK_COLOR = (256, 256, 256)
    color_schemes = st.sidebar.radio("Choose your color scheme", ("Red", "Blue", "Black", "Pink","Lavender"))
    if color_schemes == "Red":
        BG_COLOR = (255, 0, 0)
    elif color_schemes == "Blue":
        BG_COLOR = (0, 0, 255)
    elif color_schemes == "Black":
        BG_COLOR = (0, 0, 0)
    elif color_schemes == "Pink":
        BG_COLOR = (219, 112, 147)
    elif color_schemes == "Lavender":
        BG_COLOR =(230, 230, 250)
    image_file_path = st.sidebar.file_uploader("Upload file")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        results = model_selfie.process(image)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        fg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        output_image = np.where(condition, image, bg_image)
        st.image(output_image)
import numpy as np
from tensorflow import keras
import argparse
import cv2
import os

def predict_faces(img, faces, model, preprocess_img=keras.applications.mobilenet_v2.preprocess_input):
    faces_processed = []
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = keras.preprocessing.image.img_to_array(face)
        face = preprocess_img(face)
        faces_processed.append(face)
    return model(np.array(faces_processed))

def detect_faces(img, cascade):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_img, 1.1, 4)
    return faces

def display_predictions(img, preds, faces):
    for (pred, (x, y, w, h)) in zip(preds, faces):
        if pred < 0.5:
            label = 'Mask'
            confidence = 1 - pred[0]
            color = (0, 255, 0)
        else:
            label = 'No mask'
            confidence = pred[0]
            color = (0, 0, 255)
        percent = confidence * 100
        text = f"{label} {percent:.2f}%"
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)


def main():
    main_dir = os.getcwd()
    model_path = os.path.join(main_dir, 'face_mask_detection_model')
    model = keras.models.load_model(model_path)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-i", "--image", required=False, help='path to input image')
    args = vars(arg_parse.parse_args())
    if args['image']:
        image_path = args['image']
        image = cv2.imread(image_path)
        faces = detect_faces(image, face_cascade)
        preds = predict_faces(image, faces, model)
        display_predictions(image, preds, faces)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cam_feed = cv2.VideoCapture(0)
    while True:
        _, img = cam_feed.read()
        faces = detect_faces(img, face_cascade)
        if len(faces) > 0:
            preds = predict_faces(img, faces, model)
            display_predictions(img, preds, faces)
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cam_feed.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

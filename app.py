from flask import Flask,jsonify,request

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
import tensorflow as tf
from keras.preprocessing import image
import pandas as pd

import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
import cv2


import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
# from google.cloud import storage


app = Flask(__name__)
cred = credentials.Certificate("service_account_credentials.json")
fire_app = firebase_admin.initialize_app(cred, { 'storageBucket' : 'myprojects-92838.appspot.com' })


@app.route('/')
def index():
    return 'Successfully Working'


@app.route('/detect')
def app_abhi():
    # Set image dimensions
    img_height = 180
    img_width = 180

    # Load the pre-trained model
    model = tf.keras.models.load_model('Mango_Fruit_Detection_Model.h5')

    # Class names and mapping
    class_names = ['Alternaria', 'Anthracnose', 'Black_Mould_Rot', 'Healthy', 'Stem_and_Rot']

    # Read the dataset
    df = pd.read_csv("mango_dataset/Mango_fruit_dataset.csv")
    df = df.replace('', np.nan)
    df = df.dropna(axis="columns", how="any")

    # Image path
    path = "mango_dataset/img/healty_img.jpg"


    # Load image from Cloud Storage
    bucket = storage.bucket()
    blob = bucket.get_blob("detect_fruit.jpg")  # blob
    blob = bucket.blob("detect_fruit.jpg")
    blob.download_to_filename("test/download_file.jpg")
    arr = np.frombuffer(blob.download_as_string(), np.uint8)  # array of bytes
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # actual image
    img = cv2.resize(img, (img_height, img_width))
    cv2.imshow("image",img)

    # Preprocess the image
    # img = preprocess_input(img)
    # img = np.expand_dims(img, axis=0)

    import os

    # Specify the directory path and file name
    directory_path = "test/"
    file_name = "download_file.jpg"

    # Construct the full file path
    path = os.path.join(directory_path, file_name)

    # Check if the file exists before further processing
    if os.path.exists(path):
        print(f"The file path is: {path}")
        # Your further processing logic here
    else:
        print(f"The file {path} does not exist.")


    #something
    img = image.load_img(path,target_size=(img_height,img_width))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)


    # Make predictions
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    print(f"Predicted class: {predicted_class}")
    print(f"Predict Disease: {class_names[predicted_class]}")

    # Extract information from the dataset
    Sn_no, Disease_Type, Severity, Location_Date, Description, Symptoms, Diagnosis, Precautions = df.loc[predicted_class, :]

    print(f"\n\nDisease Type: {Disease_Type}\nSeverity: {Severity}\nDescription: {Description}\nSymptoms: {Symptoms}\nDiagnosis: {Diagnosis}\nPrecautions: {Precautions}")

    path = os.path.abspath(path)
    # Create a dictionary with the dataset information
    dataset = {
        "Disease_Type": Disease_Type,
        "Severity": Severity,
        "Description": Description,
        "Symptoms": Symptoms,
        "Diagnosis": Diagnosis,
        "Precautions": Precautions
    }

    return dataset

if __name__ == '__main__':
   app.run(debug=True ,host='0.0.0.0',port=5000)
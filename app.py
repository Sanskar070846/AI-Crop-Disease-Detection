from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

app = Flask(__name__)

model = tf.keras.models.load_model("model/crop_model.h5")

with open("model/class_names.json","r") as f:
    class_names = json.load(f)


# Treatment suggestions
treatments = {
"Apple___Apple_scab":"Use fungicide sprays and remove infected leaves.",
"Apple___Black_rot":"Prune infected branches and apply fungicide.",
"Apple___healthy":"Your plant is healthy. No treatment required.",
"Potato___Early_blight":"Use copper fungicide and remove infected leaves.",
"Potato___Late_blight":"Apply fungicide immediately and avoid overhead watering.",
"Tomato___Early_blight":"Use fungicide and improve air circulation.",
"Tomato___Late_blight":"Remove infected leaves and apply fungicide.",
"Tomato___healthy":"Plant is healthy."
}


def preprocess(img):

    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    return img


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    img = Image.open(file)

    processed_img = preprocess(img)

    prediction = model.predict(processed_img)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))*100

    # clean disease name
    clean_name = predicted_class.split("___")[1].replace("_"," ")
    
    if "healthy" in predicted_class.lower():
        treatment = "Plant is healthy. No treatment required."
    else:
        treatment = treatments.get(predicted_class,"Consult agricultural expert for treatment.")

    return render_template(
        "index.html",
        prediction=clean_name,
        confidence=round(confidence,2),
        treatment=treatment
    )


if __name__ == "__main__":
    app.run(debug=True)
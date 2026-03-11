from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/crop_model.h5")

# Load class names
with open("model/class_names.json","r") as f:
    class_names = json.load(f)

# Load advisory dataset
with open("model/disease_advisory.json","r") as f:
    advisory = json.load(f)


# Image preprocessing
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


    # Clean disease name for UI
    clean_name = predicted_class.replace("___"," ").replace("_"," ")


    # Healthy condition
    if "healthy" in predicted_class.lower():

        cause = "No disease detected."
        treatment = "Plant is healthy. No treatment required."
        pesticides = []

    else:

        info = advisory.get(predicted_class,{})

        cause = info.get("cause","Information not available")
        treatment = info.get("treatment","Consult agriculture expert")
        pesticides = info.get("pesticides",[])


    return render_template(
        "index.html",
        prediction=clean_name,
        confidence=round(confidence,2),
        cause=cause,
        treatment=treatment,
        pesticides=pesticides
    )


if __name__ == "__main__":
    app.run(debug=True)
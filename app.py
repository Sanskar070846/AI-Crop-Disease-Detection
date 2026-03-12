import email
from unicodedata import name

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from weather_api import get_weather
import sqlite3

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

@app.route("/scan")
def scan():
    return render_template("scan_crop.html")

@app.route("/crops")
def crops():
    return render_template("crops.html")


@app.route("/insights")
def insights():
    return render_template("insights.html")


@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("farmers.db")
        cursor = conn.cursor()

        cursor.execute(
        "SELECT * FROM farmers WHERE email=? AND password=?",
        (email,password)
        )

        user = cursor.fetchone()

        conn.close()

        if user:
            return render_template("index.html",user=user[1])
        else:
            return render_template("login.html",error="Invalid credentials")

    return render_template("login.html")

@app.route("/weather", methods=["GET","POST"])
def weather():

    if request.method == "POST":

        city = request.form["city"]

        data = get_weather(city)

        return render_template(
            "weather.html",
            temp=data["temperature"],
            feels=data["feels_like"],
            humidity=data["humidity"],
            wind=data["wind"],
            description=data["description"],
            icon=data["icon"],
            city=city
        )

    return render_template("weather.html")

@app.route("/signup", methods=["GET","POST"])
def signup():

    if request.method == "POST":

        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")

        conn = sqlite3.connect("farmers.db")
        cursor = conn.cursor()

        cursor.execute(
        "INSERT INTO farmers(name,email,password) VALUES(?,?,?)",
        (name,email,password)
        )

        conn.commit()
        conn.close()

        return render_template("login.html",message="Account created. Please login.")

    return render_template("signup.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    img = Image.open(file).convert("RGB")

    processed_img = preprocess(img)

    prediction = model.predict(processed_img)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    # Clean name
    clean_name = predicted_class.replace("___", " ").replace("_", " ")

    if "healthy" in predicted_class.lower():

        cause = "No disease detected."
        treatment = "Plant is healthy. No treatment required."
        pesticides = []

    else:

        info = advisory.get(predicted_class, {})

        cause = info.get("cause", "Information not available")
        treatment = info.get("treatment", "Consult agriculture expert")
        pesticides = info.get("pesticides", [])

    # IMPORTANT: Return scan_crop.html instead of index.html
    return render_template(
        "scan_crop.html",
        prediction=clean_name,
        confidence=round(confidence, 2),
        cause=cause,
        treatment=treatment,
        pesticides=pesticides
    )


if __name__ == "__main__":
    app.run(debug=True)
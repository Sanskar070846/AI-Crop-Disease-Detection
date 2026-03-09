import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("model/crop_model.h5")

# Load class names
with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

# Image to test
img_path = "test.jpg"

# Load image
img = image.load_img(img_path, target_size=(224,224))

img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print("\nPrediction:", predicted_class)
print("Confidence:", round(confidence,2), "%")
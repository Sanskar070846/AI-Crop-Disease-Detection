from PIL import Image
import os

dataset = "dataset/color"

for root, dirs, files in os.walk(dataset):
    for file in files:
        path = os.path.join(root, file)
        try:
            img = Image.open(path)
            img.verify()
        except:
            print("Removing corrupted:", path)
            os.remove(path)

print("Dataset cleaned successfully")
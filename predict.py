from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import io

app = Flask(__name__)

# Load model
model = load_model("best_model_tf17.h5")

# Define labels
class_labels = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

def preprocess_image(file):

    img = image.load_img(io.BytesIO(file.read()))
    img_array = image.img_to_array(img)

    img_array = tf.image.central_crop(img_array, central_fraction=0.8) 
    img_array = tf.image.resize(img_array, (150, 150))

    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0

    return img_array


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"Error": "Please upload picture"}), 400

    file = request.files["file"]
    img_array = preprocess_image(file) 

    predictions = model.predict(img_array) #use model
    predicted_class = class_labels[np.argmax(predictions)] #use BentoML API, get the index of maximum value to map the class
    confidence = float(np.max(predictions))

    return jsonify({"class": predicted_class, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)

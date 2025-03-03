from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("my_model.h5")  # Ensure this path is correct

# Class labels
class_labels = ["Citrus Canker", "Healthy", "Melanose"]

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure an image was uploaded
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        # Read and process the image
        image_file = request.files["image"]
        img = Image.open(io.BytesIO(image_file.read()))
        processed_img = preprocess_image(img)

        # Make a prediction
        predictions = model.predict(processed_img)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_class = class_labels[predicted_index]
        confidence = float(np.max(predictions))

        return jsonify({"class": predicted_class, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the server
if __name__ == "__main__":
    app.run(debug=True)

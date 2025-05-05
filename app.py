from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model/best_model_transfer_learning.keras', compile=False)

# Read accuracy and loss from saved metrics file
metrics = {"val_accuracy": "N/A", "val_loss": "N/A"}
try:
    with open("model/metrics.txt", "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            metrics[key] = f"{float(value) * 100:.2f}%" if "accuracy" in key else value
except FileNotFoundError:
    print("metrics.txt not found.")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    background_image = url_for('static', filename='images/background.jpg')  # Get image path for background
    if request.method == "POST":
        file = request.files["file"]
        if file:
            # Save the uploaded image
            file_path = "static/uploaded_image.jpg"
            file.save(file_path)

            # Preprocess the image
            img = image.load_img(file_path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction_value = model.predict(img_array)[0][0]

            # Determine if it's a dangerous or non-dangerous animal
            if prediction_value > 0.5:
                prediction = "Non-Dangerous Animal"
            else:
                prediction = "Dangerous Animal"

            # Pass prediction and metrics to the template
            return render_template("index.html", prediction=prediction, metrics=metrics, background_image=background_image)

    return render_template("index.html", prediction=prediction, metrics=metrics, background_image=background_image)

if __name__ == "__main__":
    app.run(debug=True)
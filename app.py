from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('best_model_transfer_learning.keras', compile=False)

# Define the path to the class labels
class_labels = os.listdir('data')  # This assumes your classes are directly under 'data'

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    show_image = False

    if request.method == "POST":
        if "camera" in request.form and "captured_image" in request.form:
            # Handle captured camera image (base64)
            image_data = request.form.get("captured_image", "")

            if "," in image_data:
                image_data = image_data.split(",")[1]  # Remove base64 prefix
                image_bytes = base64.b64decode(image_data)

                image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                image_pil = image_pil.resize((128, 128))
                image_pil.save("static/uploaded_image.jpg")

                img_array = np.array(image_pil) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
            else:
                prediction = "No image data received from camera."
                return render_template("index.html", prediction=prediction, show_image=False)


            image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
            image_pil = image_pil.resize((128, 128))
            image_pil.save("static/uploaded_image.jpg")

            img_array = np.array(image_pil) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        else:
            # Handle file upload
            file = request.files["file"]
            if file:
                file_path = "static/uploaded_image.jpg"
                file.save(file_path)

                img = image.load_img(file_path, target_size=(128, 128))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_labels[predicted_class_index]
        prediction = f"This is a {predicted_class}."

        show_image = True

    return render_template("index.html", prediction=prediction, show_image=show_image)

if __name__ == "__main__":
    app.run(debug=True)

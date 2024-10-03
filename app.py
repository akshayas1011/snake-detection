from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploader/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
SIZE = 24
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST', 'GET'])
def Upload():
    if request.method == 'POST':
        file = request.files['image']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], '1.png'))

        model = keras.models.load_model(r'model\model1.h5')
        categories = ['Banded Racer', 'Checkered Keelback', 'Green Tree Vine', 'Common Rat Snake', 'Common Krait',
                      'King Cobra', 'Spectacled Cobra']

        nimage = cv2.imread(os.path.join("static", "uploader", "1.png"), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(nimage, (SIZE, SIZE))
        image = image / 255.0
        prediction = model.predict(np.array(image).reshape(-1, SIZE, SIZE, 1))
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = categories[predicted_class_index]

        # Check if the predicted class is venomous or non-venomous
        venomous_classes = ['King Cobra', 'Spectacled Cobra','Common Krait']
        if predicted_class_label in venomous_classes:
            venomous_status = "Venomous"
        else:
            venomous_status = "Non-Venomous"

        return render_template('result.html', value=predicted_class_label, venomous_status=venomous_status)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

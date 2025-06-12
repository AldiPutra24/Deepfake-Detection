import os
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow.keras.preprocessing import image
from model import build_model, convert_to_ela_image
from PIL import Image

app = Flask(__name__)

# Bangun arsitektur dan muat bobot
model = build_model()
model.load_weights('model/best_model_converted.h5')
class_labels = ['fake', 'real']

# Buat folder jika belum ada
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/ela', exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None
    ela_filename = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            filename = file.filename
            filepath = os.path.join('static/uploads', filename)
            file.save(filepath)

            # Konversi ke ELA
            ela_arr = convert_to_ela_image(filepath, quality=90, resize_to=(224, 224))

            # Simpan hasil ELA sebagai file
            ela_image_pil = Image.fromarray((ela_arr * 255).astype(np.uint8))
            ela_filename = filename  # Bisa diganti kalau mau beda nama
            ela_filepath = os.path.join('static/ela', ela_filename)
            ela_image_pil.save(ela_filepath)

            # Prediksi
            img_batch = np.expand_dims(ela_arr, axis=0)
            pred = model.predict(img_batch)[0]
            idx = np.argmax(pred)
            prediction = class_labels[idx]
            confidence = round(pred[idx] * 100, 2)
            print('Softmax:', pred, '->', prediction)

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        filename=filename,
        ela_filename=ela_filename,
        uploaded=bool(prediction)
    )

if __name__ == '__main__':
    app.run(debug=True)
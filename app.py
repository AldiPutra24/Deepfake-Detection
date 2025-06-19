import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing import image
from model import build_model, convert_to_ela_image
from PIL import Image

app = Flask(__name__)

# Load model dan bobot
model = build_model()
model.load_weights('model/best_model_converted.h5')
class_labels = ['fake', 'real']

# Pastikan folder untuk upload dan ELA tersedia
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/ela', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('static/uploads', filename)
        file.save(filepath)

        # Konversi ke ELA dan simpan
        ela_array = convert_to_ela_image(filepath, quality=90, resize_to=(224, 224))
        ela_image = Image.fromarray((ela_array * 255).astype(np.uint8))
        ela_path = os.path.join('static/ela', filename)
        ela_image.save(ela_path)

        # Prediksi menggunakan model CNN
        img_batch = np.expand_dims(ela_array, axis=0)
        prediction_probs = model.predict(img_batch)[0]
        predicted_index = np.argmax(prediction_probs)
        prediction_label = class_labels[predicted_index]
        confidence = round(prediction_probs[predicted_index] * 100, 2)

        return jsonify({
            'success': True,
            'filename': filename,
            'prediction': prediction_label,
            'confidence': confidence
        })
    return jsonify({'success': False, 'message': 'Invalid file format or no file uploaded'})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

def secure_filename(filename):
    return filename.replace(" ", "_")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

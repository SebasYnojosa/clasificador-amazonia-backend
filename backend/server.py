from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import flask_cors as cors

app = Flask(__name__)
cors.CORS(app)

UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'public/modelo/first_modelv5.keras'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASSES = ['animal', 'humano', 'objeto', 'paisaje']

# Asegúrate de que la carpeta de subidas exista
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar el modelo preentrenado
model = tf.keras.models.load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print('No file uploaded.')
        return 'No file uploaded.', 400

    file = request.files['image']

    if file.filename == '':
        print('No file uploaded.')
        return 'No file uploaded.', 400

    if not allowed_file(file.filename):
        print('File type not allowed. Only .jpg, .jpeg, and .png are allowed.')
        return 'File type not allowed. Only .jpg, .jpeg, and .png are allowed.', 400

    # Guardar el archivo subido
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Cargar y preprocesar la imagen
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Hacer la predicción
        prediction = model.predict(img_array)
        prediction_list = prediction.tolist()

        # Eliminar el archivo temporal
        os.remove(file_path)

        predicted_class = CLASSES[np.argmax(prediction_list[0])]

        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return f'Error processing image: {str(e)}', 500

if __name__ == '__main__':
    app.run(port=3000)
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import flask_cors as cors

app = Flask(__name__)
cors.CORS(app)

UPLOAD_FOLDER = 'uploads/first_model'

FIRST_MODEL_PATH = 'public/modelos/first_modelv5.keras'
SECOND_MODEL_PATH = 'public/modelos/second_modelv69_definitive.keras'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASSES = ['animal', 'humano', 'objeto inanimado', 'paisaje']

# Asegúrate de que la carpeta de subidas exista
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar el modelo preentrenado
first_model = tf.keras.models.load_model(FIRST_MODEL_PATH)
second_model = tf.keras.models.load_model(SECOND_MODEL_PATH)

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
        prediction = first_model.predict(img_array)
        prediction_list = prediction.tolist()

        predicted_class = CLASSES[np.argmax(prediction_list[0])]

        if predicted_class != 'humano':
            # Eliminar el archivo temporal
            os.remove(file_path)

            return jsonify({'prediction': predicted_class}), 200
        else:
            # Cargar y preprocesar la imagen
            img = Image.open(file_path)
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Hacer la predicción
            prediction = second_model.predict(img_array)
            prediction_list = prediction.tolist()

            feature_names = ['pendant', 'corporal_paint', 'face_paint', 'modern_clothing', 'creole_clothing',
                        'ancestral_clothing', 'animal_fur', 'feathers', 'hat', 'nose_piercing',
                        'bowl_cut', 'tendrils', 'arm_accesory', 'bracelets']
            
            threshold = 0.50

            # Crear un diccionario con los nombres de las características
            features_dict = {name: (1 if pred >= threshold else 0) for name, pred in zip(feature_names, prediction_list[0])}

            # Contar las características importantes presentes para Akawayo
            important_akawayo_features = ['pendant', 'face_paint', 'creole_clothing', 'hat']
            akawayo_feature_count = sum(features_dict.get(feature, 0) for feature in important_akawayo_features)

            # Contar las características importantes presentes para Karina
            important_karina_features = ['ancestral_clothing', 'face_paint', 'bracelets']
            karina_feature_count = sum(features_dict.get(feature, 0) for feature in important_karina_features)

            # Contar las caracteristicas importantes presentes para Arawak
            important_arawak_features = ['pendant', 'face_paint', 'modern_clothing', 'ancestral_clothing', 'hat', 'bracelets']
            arawak_feature_count = sum(features_dict.get(feature, 0) for feature in important_arawak_features)

            # Contar las caracteristicas importantes presentes para E'ñepa
            important_enepa_features = ['pendant', 'face_paint', 'creole_clothing', 'ancestral_clothing', 'bracelets']
            enepa_feature_count = sum(features_dict.get(feature, 0) for feature in important_enepa_features)

            # Contar las caracteristicas importantes presentes para Mapoyo
            important_mapoyo_features = ['pendant', 'corporal_paint', 'modern_clothing', 'ancestral_clothing', 'animal_fur', 'arm_accesory', 'bracelets']
            mapoyo_feature_count = sum(features_dict.get(feature, 0) for feature in important_mapoyo_features)

            # Contar las caracteristicas importantes presentes para Yabarana
            important_yabarana_features = ['modern_clothing', 'creole_clothing', 'ancestral_clothing']
            yabarana_feature_count = sum(features_dict.get(feature, 0) for feature in important_yabarana_features)

            # Contar las caracteristicas importantes presentes para Jivi
            important_jivi_features = ['face_paint', 'modern_clothing', 'hat']
            jivi_feature_count = sum(features_dict.get(feature, 0) for feature in important_jivi_features)

            # Contar las caracteristicas importantes presentes para Jodi
            important_jodi_features = ['pendant', 'ancestral_clothing']
            jodi_feature_count = sum(features_dict.get(feature, 0) for feature in important_jodi_features)

            # Contar las caracteristicas importantes presentes para Pemón
            important_pemon_features = ['pendant', 'corporal_paint', 'face_paint', 'modern_clothing', 'ancestral_clothing']
            pemon_feature_count = sum(features_dict.get(feature, 0) for feature in important_pemon_features)

            # Contar las caracteristicas importantes presentes para Puinave
            important_puinave_features = ['modern_clothing']
            puinave_feature_count = sum(features_dict.get(feature, 0) for feature in important_puinave_features)

            # Contar las caracteristicas importantes presentes para Piaroa
            important_piaroa_features = ['pendant', 'modern_clothing', 'ancestral_clothing']
            piaroa_feature_count = sum(features_dict.get(feature, 0) for feature in important_piaroa_features)

            # Contar las caracteristicas importantes presentes para Warao
            important_warao_features = ['pendant', 'modern_clothing', 'ancestral_clothing']
            warao_feature_count = sum(features_dict.get(feature, 0) for feature in important_warao_features)

            # Contar las caracteristicas importantes presentes para Yanomami
            important_yanomami_features = ['pendant', 'corporal_paint', 'ancestral_clothing', 'nose_piercing', 'bowl_cut', 'tendrils', 'arm_accesory']
            yanomami_feature_count = sum(features_dict.get(feature, 0) for feature in important_yanomami_features)

            # Contar las caracteristicas importantes presentes para Ye'kwana
            important_yekwana_features = ['modern_clothing', 'ancestral_clothing']
            yekwana_feature_count = sum(features_dict.get(feature, 0) for feature in important_yekwana_features)

            # Contar las caracteristicas importantes presentes para No Etnia
            important_noetnia_features = ['modern_clothing']
            noetnia_feature_count = sum(features_dict.get(feature, 0) for feature in important_noetnia_features)

            # Definir funciones para calcular el valor de pertenencia
            def calculate_membership_akawayo(count):
                if count == 0:
                    return 1.99
                elif count == 1:
                    return 24.99
                elif count >= 2 and count <= 3:
                    return 64.50
                elif count == 4:
                    return 84.99

            def calculate_membership_karina(count):
                if count == 0:
                    return 1.99
                elif count == 1:
                    return 24.99
                elif count == 2:
                    return 64.50
                elif count == 3:
                    return 84.99

            def calculate_membership_arawak(count):
                if count == 0:
                    return 1.99
                elif count >= 1 and count <= 2:
                    return 30.99
                elif count >= 3 and count <= 5:
                    return 68.50
                elif count == 6:
                    return 83.99

            def calculate_membership_enepa(count):
                if count == 0:
                    return 1.99
                elif count == 1:
                    return 24.99
                elif count >= 2 and count <= 4:
                    return 64.50
                elif count == 5:
                    return 84.99

            def calculate_membership_mapoyo(count):
                if count == 0:
                    return 1.99
                elif count >= 1 and count <= 2:
                    return 30.99
                elif count >= 3 and count <= 6:
                    return 70.50
                elif count == 7:
                    return 86.99

            def calculate_membership_yabarana(count):
                if count == 0:
                    return 1.99
                elif count == 1:
                    return 20.99
                elif count == 2:
                    return 54.99
                elif count == 3:
                    return 80.99

            def calculate_membership_jivi(count):
                if count == 0:
                    return 1.99
                elif count == 1:
                    return 20.99
                elif count == 2:
                    return 54.99
                elif count == 3:
                    return 80.99

            def calculate_membership_jodi(count):
                if count == 0:
                    return 1.99
                elif count == 1:
                    return 20.99
                elif count == 2:
                    return 54.99
                elif count == 3:
                    return 80.99

            def calculate_membership_pemon(count):
                if count == 0:
                    return 1.99
                elif count == 1:
                    return 24.99
                elif count >= 2 and count <= 4:
                    return 64.50
                elif count == 5:
                    return 84.99

            def calculate_membership_puinave(count):
                if count == 0:
                    return 1.99
                elif count == 1:
                    return 24.50

            def calculate_membership_piaroa(count):
                if count == 0:
                    return 1.99
                elif count == 1:
                    return 20.99
                elif count == 2:
                    return 54.99
                elif count == 3:
                    return 80.99

            def calculate_membership_warao(count):
                if count == 0:
                    return 1.99
                elif count == 1:
                    return 20.99
                elif count == 2:
                    return 54.99
                elif count == 3:
                    return 80.99

            def calculate_membership_yanomami(count):
                if count == 0:
                    return 1.99
                elif count >= 1 and count <= 2:
                    return 30.99
                elif count >= 3 and count <= 6:
                    return 70.50
                elif count == 7:
                    return 86.99

            def calculate_membership_yekwana(count):
                if count == 0:
                    return 1.99
                elif count == 1:
                    return 34.99
                elif count == 2:
                    return 75.99

            def calculate_membership_noetnia(count, features):
                if count == 1 and features.get('modern_clothing') == 1: # Asegurarse de que todos los demás elementos son 0
                    for key, value in features.items():
                        if key != 'modern_clothing' and value != 0:
                            return 9.99 # Regresar 0 si cualquier otro elemento tiene un valor diferente de 0
                    return 77.99 # Si todos los demás elementos son 0, regresar 77.99
                return 1.99 # Regresar 0 si la condición no se cumple

            # Crear el diccionario de pertenencia
            membership_dict = {
                'Akawayo': calculate_membership_akawayo(akawayo_feature_count),
                'Karina': calculate_membership_karina(karina_feature_count),
                'Arawak': calculate_membership_arawak(arawak_feature_count),
                'Enepa': calculate_membership_enepa(enepa_feature_count),
                'Mapoyo': calculate_membership_mapoyo(mapoyo_feature_count),
                'Yabarana': calculate_membership_yabarana(yabarana_feature_count),
                'Jivi': calculate_membership_jivi(jivi_feature_count),
                'Jodi': calculate_membership_jodi(jodi_feature_count),
                'Pemon': calculate_membership_pemon(pemon_feature_count),
                'Puinave': calculate_membership_puinave(puinave_feature_count),
                'Piaroa': calculate_membership_piaroa(piaroa_feature_count),
                'Warao': calculate_membership_warao(warao_feature_count),
                'Yanomami': calculate_membership_yanomami(yanomami_feature_count),
                'Yekwana': calculate_membership_yekwana(yekwana_feature_count),
                'NoEtnia': calculate_membership_noetnia(noetnia_feature_count, features_dict)
            }


            # Obtener los 5 elementos con los valores más altos
            top_5_memberships = dict(sorted(membership_dict.items(), key=lambda item: item[1], reverse=True)[:5])

            print(top_5_memberships)

            # Eliminar el archivo temporal
            os.remove(file_path)

            return jsonify(top_5_memberships), 200
    except Exception as e:
        return f'Error processing image: {str(e)}', 500

if __name__ == '__main__':
    app.run(port=3000)
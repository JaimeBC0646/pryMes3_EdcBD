from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('SVC_Model.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('predRendimiento.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        materia1 = float(request.form['materia1'])
        materia2 = float(request.form['materia2'])
        materia2 = float(request.form['materia2'])
        materia3 = float(request.form['materia3'])
        materia4 = float(request.form['materia4'])
        materia5 = float(request.form['materia5'])
        materia6 = float(request.form['materia6'])
        materia7 = float(request.form['materia7'])
        materia8 = float(request.form['materia8'])
        
        
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[materia1, materia2, materia3, materia4, materia5, materia6, materia7, materia8]],
         columns=['ESPANOL', 'MATEMATICAS', 'INGLES', 'CIENCIAS_NATURALES', 'HISTORIA', 'EDUCACION_FISICA', 'ARTES', 'TECNOLOGIA'])

        app.logger.debug(f'DataFrame creado: {data_df}')
        
        #scaler = StandardScaler()
        #df_scaled = scaler.fit_transform(data_df)
        #df_scaled = pd.DataFrame(df_scaled, columns = data_df.columns)
        #app.logger.debug(f'DataFrame escalad: {df_scaled}')


        # Realizar pyredicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        prom = (materia1 + materia2 + materia3 + materia4 + materia5 + materia6 + materia7 + materia8) / 8
        # Convertir la predicción a un tipo de datos nativo de Python
        #int_Prediction = int(prediction[0])
        
        
        #if int_Prediction == 0:
        #    prediction_result = "Excelente"
        #elif int_Prediction == 1:
        #    prediction_result = "Bueno"
        #elif int_Prediction == 2:
        #    prediction_result = "Regular"
        #elif int_Prediction == 3:
        #    prediction_result = "Malo"
        
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediction[0], 'prom':prom})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)


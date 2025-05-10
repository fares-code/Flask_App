from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes


model = joblib.load('Model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # استلام البيانات المدخلة (الـ JSON)
    data = request.json
    
    # استخراج البيانات من الـ JSON
    input_data = [
        data['HighBP'],
        data['HighChol'],
        data['CholCheck'],
        data['BMI'],
        data['Smoker'],
        data['Stroke'],
        data['HeartDiseaseorAttack'],
        data['Veggies'],
        data['HvyAlcoholConsump'],
        data['AnyHealthcare'],
        data['NoDocbcCost'],
        data['GenHlth'],
        data['MentHlth'],
        data['PhysHlth'],
        data['Sex'],
        data['Age'],
        data['Education'],
        data['Income']
    ]
    
    # تحويل البيانات إلى مصفوفة NumPy
    input_features = np.array(input_data).reshape(1, -1)  # تحويل المدخلات إلى مصفوفة صف واحد
    
    # تنفيذ التنبؤ باستخدام النموذج
    prediction = model.predict(input_features)
    print("Prediction",prediction)
    # إرجاع النتيجة في شكل JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

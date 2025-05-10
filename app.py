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
   
    data = request.json
    
   
    input_data = [
        data['HighBP'],               # HighBP
        data['HighChol'],             # HighChol
        data['CholCheck'],            # CholCheck
        data['BMI'],                  # BMI
        data['Smoker'],               # Smoker
        data['Stroke'],               # Stroke
        data['HeartDiseaseorAttack'], # HeartDiseaseorAttack
        data['Veggies'],              # Veggies
        data['HvyAlcoholConsump'],    # HvyAlcoholConsump
        data['AnyHealthcare'],        # AnyHealthcare
        data['NoDocbcCost'],          # NoDocbcCost
        data['GenHlth'],              # GenHlth
        data['MentHlth'],             # MentHlth
        data['PhysHlth'],             # PhysHlth
        data['Sex'],                  # Sex
        data['Age'],                  # Age
        data['Education'],            # Education
        data['Income']                # Income
    ]
    
   
    input_features = np.array(input_data).reshape(1, -1)  
    
    
    prediction = model.predict(input_features)
    
   
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

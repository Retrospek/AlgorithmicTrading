from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return 'Welcome to the Prediction API!'
    
@app.route('/predict', methods=['POST']) 
def predict():
    try:
        data = request.get_json()
        
        features = np.array(data['features']).reshape(1, -1)  

        prediction = model.predict(features)

        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model
with open('trend_predictor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return "Micro-Trend Predictor API is Live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'features' not in data:
        return jsonify({'error': 'Missing "features" key in request'}), 400

    try:
        features = np.array(data['features']).reshape(1, -1)  # Convert to array
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

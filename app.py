from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open("trend_predictor.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return jsonify({"message": "API is working!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]
        prediction = model.predict([data])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run()

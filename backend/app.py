from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from ddos_model import evaluate_model  # Import your model function

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    df = pd.read_csv(file)
    
    accuracy = evaluate_model(df)  # Call ML model function
    
    return jsonify({'accuracy': accuracy})

if __name__ == '__main__':
    app.run(debug=True)

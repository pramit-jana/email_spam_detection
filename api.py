from flask import Flask, request, jsonify
from data_process_method import data_process
from mlflow.tracking import MlflowClient
import mlflow
import joblib
import os

# Initialize Flask application
app = Flask(__name__)

# Set the MLflow tracking URI to use HTTP
mlflow.set_tracking_uri("http://localhost:5000")  # Replace with the appropriate URI of your MLflow tracking server
client = MlflowClient()
tmp_path = client.download_artifacts(run_id="29c5d8673ba94829ac95a5f2a4fbb94c", path='model/model.pkl')
model = joblib.load(os.path.join(".", tmp_path))

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request...")
    # Get input data from request
    data = request.get_json()
    print("Request data:", data)
    email = data['email']
    
    # Perform data preprocessing
    input_email = data_process(email)

    prediction = model.predict([input_email])[0]

    # Return the prediction
    print("Sending response...")

    if prediction==0:
        res="ham"
    else:
        res="spam"

    return jsonify({'prediction': res})

if __name__ == '__main__':
    app.run(debug=True,port=8080)  # You can set debug to False in production

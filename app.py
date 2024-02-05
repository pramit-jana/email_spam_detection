from flask import Flask, request, jsonify,render_template
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

@app.route('/', methods=['GET','POST'])
def predict():
    res = None
    if request.method=='POST':
        email=request.form.get('inputTxt')
        process_email = data_process(email)
        prediction = model.predict([process_email])[0]
        if prediction==0:
            res="ham"
        else:
            res="spam"


    return render_template('index.html',result=res)

if __name__ == '__main__':
    app.run(debug=True,port=5002)  # You can set debug to False in production

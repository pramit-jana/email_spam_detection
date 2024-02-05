from mlflow.tracking import MlflowClient
import pickle
import mlflow
# Set the MLflow tracking URI to use HTTP
mlflow.set_tracking_uri("http://localhost:5000")  # Replace with the appropriate URI of your MLflow tracking server
client = MlflowClient()
tmp_path = client.download_artifacts(run_id="4d2d884b9faf4c6ab22c0662fe0f8555", path='model/model.pkl')
f = open(tmp_path,'rb')
model = pickle.load(f)
f.close()
client.list_artifacts(run_id="4d2d884b9faf4c6ab22c0662fe0f8555", path="")
client.list_artifacts(run_id="4d2d884b9faf4c6ab22c0662fe0f8555", path="model")
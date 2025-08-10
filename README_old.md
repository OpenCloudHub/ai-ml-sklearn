# ai-ml-sklearn

Traditional machine learning workflows with scikit-learn demonstrating basic MLOps patterns

# Start MLflow with 0.0.0.0 to make it accessible from Docker containers

mlflow server --host 0.0.0.0 --port 8081

# Then in another terminal:

export MLFLOW_TRACKING_URI=http://172.17.0.1:8081
export MLFLOW_EXPERIMENT_NAME=wine-quality

after pusjing container, u can run:

docker build -t opencloudhuborg/wine-classifier:latest .

uv run src/training.py
uv run src/hyperparameter_tuning.py

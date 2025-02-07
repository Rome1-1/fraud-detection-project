import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

# Example to log a model with MLflow
def log_model():
    # Train a model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Log the model
    mlflow.sklearn.log_model(model, "logistic_regression_model")

log_model()


import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score,roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_score
from dotenv import find_dotenv, load_dotenv
import mlflow
import mlflow.pyfunc

# Create evaluation function
def evaluate_model(model, X, y, X_test, y_test):
    # Evaluate the model using cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    cv_score = np.mean(cv_scores)
    
    # Get the model predictions and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate the evaluation metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision= average_precision_score(y_test, y_proba)
    accuracy= accuracy_score(y_test, y_pred)
    precision= precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    f1= f1_score(y_test, y_pred)
    
    # Return evaluation metrics
    return cv_score,roc_auc,avg_precision,accuracy,precision,recall,f1
    
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Create prediction function
def mood_prediction(music):
        
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
        MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
        mlflow.set_tracking_uri("https://dagshub.com/inouyewilliam/Master-Thesis.mlflow")
    
        logged_model = 'runs:/5cf7eb61d49d4df4b38bbfa2ed92cd4c/model'
    
        loaded_model = mlflow.pyfunc.load_model(logged_model)
    
        predictions = loaded_model.predict(pd.read_csv(music))
        
        return list(predictions)
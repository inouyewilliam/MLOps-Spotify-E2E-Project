
import pandas as pd
import os
import mlflow
import mlflow.pyfunc

def mood_prediction(music):
    
        os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/inouyewilliam/Master-Thesis.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = "inouyewilliam"
        os.environ["MLFLOW_TRACKING_PASSWORD"] ="b185d44c9fe85ded477875ff2ba1b4d229006006"
    
        mlflow.set_tracking_uri("https://dagshub.com/inouyewilliam/Master-Thesis.mlflow")
    
        logged_model = 'runs:/5cf7eb61d49d4df4b38bbfa2ed92cd4c/model'
    
        loaded_model = mlflow.pyfunc.load_model(logged_model)
    
        predictions = loaded_model.predict(pd.read_csv(music))
        
        return list(predictions)
import os
import warnings
import sys
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split, KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from model_deployment import evaluate_model
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


#Import the credentials to register in MLflow
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri("https://dagshub.com/inouyewilliam/Master-Thesis.mlflow")

#Data Types dictionary
dtype = {
       'popularity':'int8',
       'genres': str,
       'sub-genres': str,
       'explicit':'int8', 
       'followers': int, 
       'danceability':float,
       'energy': float, 
       'key':'int8',
       'loudness': float,
       'mode':'int8', 
       'tempo':float, 
       'duration_ms':int,
       'time_signature':'int8',
       'mood': 'int8' 
       }

#------------------- Train LightGBM model  
if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    
    # Read the final csv file
    csv = (
            "C:/Users/willi/Python/Spotify_Project/Data/final_data.csv"
        )
    try:
            data = pd.read_csv(csv, sep=",", dtype = dtype)
            
    except Exception as e:
            logger.exception(
                "Unable to download training & test CSV. Error: %s", e
            )

    # Split the data into training and test sets. (0.8, 0.2) split.
        
    X = data.drop("mood", axis=1)
    y = data["mood"]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform Feature Selection to find the best K
    
    def select_k_best(X, y, estimator, k_values=[2, 5, 7, 10, 12, 13]):
        best_k = 0
        best_score = float('-inf')
        best_selector = None
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        for k in k_values:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(k=k)),
                ("model", estimator)])
            scores = cross_val_score(pipeline, X, y, cv=cv)
            if scores.mean() > best_score:
                best_k = k
                best_score = scores.mean()
                best_selector = pipeline.named_steps["selector"]
                best_selector.fit(X, y)
                selected_features = X.columns[best_selector.get_support()]
                print(f"Best k: {best_k}")
                print(f"Selected features: {list(selected_features)}")
        return best_k
    
    estimator = LGBMClassifier()
    best_k = select_k_best(X_train, y_train, estimator, k_values=[2, 5, 7, 10, 12, 13])
    

    with mlflow.start_run():
        
        # Build a training Pipeline
     
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k= best_k)),
            ("model", LGBMClassifier())])
        
        
        # Hyperparameter Optimization
         
        param_distributions = {
        "model__max_depth": sp_randint(3, 10),
        "model__n_estimators": sp_randint(50, 200),
        "model__num_leaves": sp_randint(2, 50),
        "model__learning_rate": sp_uniform(0.001, 0.1)
        }
        
        
        random_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions, n_iter=50,
                                   cv=5, n_jobs=-1, verbose=2)
        
        random_search.fit(X_train, y_train)
        
       
                
        # Infer the model signature
        signature = infer_signature(X, random_search.predict(X))
        
        # Model Evaluation
        (cv_score,roc_auc,average_precision,accuracy,precision,recall,f1) = evaluate_model(random_search, X, y, X_test, y_test)

        print("cv_score: %s" % cv_score)
        print("best params: %s" % random_search.best_params_)
        print("roc_auc: %s" % roc_auc)
        print("average_precision: %s" % average_precision)
        print("accuracy: %s" % accuracy)
        print("precision: %s" % precision)
        print("recall: %s" % recall)
        print("f1 score: %s" % f1)

        
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("mean_cv_score", cv_score)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("average_precision", average_precision)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1 score", f1)
        
       
        # Register the model            
        mlflow.lightgbm.log_model(random_search, "model", signature = signature, registered_model_name="LgbmModel")
        

#------------------- Train ExtraTrees model
if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    
    # Read the final csv file
    csv = (
            "C:/Users/willi/Python/Spotify_Project/Data/final_data.csv"
        )
    try:
            data = pd.read_csv(csv, sep=",", dtype = dtype)
            
    except Exception as e:
            logger.exception(
                "Unable to download training & test CSV. Error: %s", e
            )

    # Split the data into training and test sets. (0.8, 0.2) split.
        
    X = data.drop("mood", axis=1)
    y = data["mood"]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    # Perform Feature Selection to find the best K
    
    def select_k_best(X, y, estimator, k_values=[2, 5, 7, 10, 12, 13]):
        best_k = 0
        best_score = float('-inf')
        best_selector = None
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        for k in k_values:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(k=k)),
                ("model", estimator)])
            scores = cross_val_score(pipeline, X, y, cv=cv)
            if scores.mean() > best_score:
                best_k = k
                best_score = scores.mean()
                best_selector = pipeline.named_steps["selector"]
                best_selector.fit(X, y)
                selected_features = X.columns[best_selector.get_support()]
                print(f"Best k: {best_k}")
                print(f"Selected features: {list(selected_features)}")
        return best_k
    
    estimator = ExtraTreesClassifier()
    best_k = select_k_best(X_train, y_train, estimator, k_values=[2, 5, 7, 10, 12, 13])
    

    with mlflow.start_run():
        
        # Build a training Pipeline
     
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k= best_k)),
            ("model", ExtraTreesClassifier())])
        
        
        # Hyperparameter Optimization
         
        param_distributions = {
        "model__n_estimators": sp_randint(50, 200),
        "model__max_depth": sp_randint(3, 10),
        "model__min_samples_split": sp_randint(2, 10),
        "model__min_samples_leaf": sp_randint(1, 10),
        "model__bootstrap": [True, False],
        "model__criterion": ["gini", "entropy"]
        }
        
        
        random_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions, n_iter=50,
                                   cv=5, n_jobs=-1, verbose=2)
        
        random_search.fit(X_train, y_train)
        
        # Infer the model signature
        signature = infer_signature(X, random_search.predict(X))
        
        # Model Evaluation
        (cv_score,roc_auc,average_precision,accuracy,precision,recall,f1) = evaluate_model(random_search, X, y, X_test, y_test)

        print("cv_score: %s" % cv_score)
        print("best params: %s" % random_search.best_params_)
        print("roc_auc: %s" % roc_auc)
        print("average_precision: %s" % average_precision)
        print("accuracy: %s" % accuracy)
        print("precision: %s" % precision)
        print("recall: %s" % recall)
        print("f1 score: %s" % f1)

        
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("mean_cv_score", cv_score)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("average_precision", average_precision)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1 score", f1)

        # Register the model           
        mlflow.sklearn.log_model(random_search, "model", signature = signature, registered_model_name="ExtraTreeModel")

#------------------- Train XGBoost model
if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    # Read the final csv file
    csv = (
            "C:/Users/willi/Python/Spotify_Project/Data/final_data.csv"
        )
    try:
            data = pd.read_csv(csv, sep=",", dtype = dtype)
            
    except Exception as e:
            logger.exception(
                "Unable to download training & test CSV. Error: %s", e
            )

    # Split the data into training and test sets. (0.8, 0.2) split.
        
    X = data.drop("mood", axis=1)
    y = data["mood"]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

       
    # Perform Feature Selection to find the best K
    
    def select_k_best(X, y, estimator, k_values=[2, 5, 7, 10, 12, 13]):
        best_k = 0
        best_score = float('-inf')
        best_selector = None
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        for k in k_values:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(k=k)),
                ("model", estimator)])
            scores = cross_val_score(pipeline, X, y, cv=cv)
            if scores.mean() > best_score:
                best_k = k
                best_score = scores.mean()
                best_selector = pipeline.named_steps["selector"]
                best_selector.fit(X, y)
                selected_features = X.columns[best_selector.get_support()]
                print(f"Best k: {best_k}")
                print(f"Selected features: {list(selected_features)}")
        return best_k
    
    estimator = XGBClassifier()
    best_k = select_k_best(X_train, y_train, estimator, k_values=[2, 5, 7, 10, 12, 13])
    

    with mlflow.start_run():
        
        # Build a training Pipeline
     
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k= best_k)),
            ("model", XGBClassifier())])
        
        
        # Hyperparameter Optimization
         
        param_distributions = {
        "model__max_depth": sp_randint(3, 10),
        "model__n_estimators": sp_randint(50, 200),
        "model__learning_rate": sp_uniform(0.001, 0.1),
        "model__subsample": sp_uniform(0.5, 0.5),
        "model__colsample_bytree": sp_uniform(0.5, 0.5),
        "model__reg_lambda": sp_uniform(0.1, 1)
        }
        
        
        random_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions, n_iter=50,
                                   cv=5, n_jobs=-1, verbose=2)
        
        random_search.fit(X_train, y_train)
        
        # Infer the model signature
        signature = infer_signature(X, random_search.predict(X))
        
        # Model Evaluation
        (cv_score,roc_auc,average_precision,accuracy,precision,recall,f1) = evaluate_model(random_search, X, y, X_test, y_test)

        print("cv_score: %s" % cv_score)
        print("best params: %s" % random_search.best_params_)
        print("roc_auc: %s" % roc_auc)
        print("average_precision: %s" % average_precision)
        print("accuracy: %s" % accuracy)
        print("precision: %s" % precision)
        print("recall: %s" % recall)
        print("f1 score: %s" % f1)

        
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("mean_cv_score", cv_score)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("average_precision", average_precision)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1 score", f1)

        # Register the model           
        mlflow.sklearn.log_model(random_search, "model",signature = signature, registered_model_name="XGBModel")

#------------------- Train Random Forest model

if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    
    # Read the final csv file
    csv = (
            "C:/Users/willi/Python/Spotify_Project/Data/final_data.csv"
        )
    try:
            data = pd.read_csv(csv, sep=",", dtype = dtype)
            
    except Exception as e:
            logger.exception(
                "Unable to download training & test CSV. Error: %s", e
            )

    # Split the data into training and test sets. (0.8, 0.2) split.
        
    X = data.drop("mood", axis=1)
    y = data["mood"]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform Feature Selection to find the best K
    
    def select_k_best(X, y, estimator, k_values=[2, 5, 7, 10, 12, 13]):
        best_k = 0
        best_score = float('-inf')
        best_selector = None
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        for k in k_values:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(k=k)),
                ("model", estimator)])
            scores = cross_val_score(pipeline, X, y, cv=cv)
            if scores.mean() > best_score:
                best_k = k
                best_score = scores.mean()
                best_selector = pipeline.named_steps["selector"]
                best_selector.fit(X, y)
                selected_features = X.columns[best_selector.get_support()]
                print(f"Best k: {best_k}")
                print(f"Selected features: {list(selected_features)}")
        return best_k
    
    estimator = RandomForestClassifier()
    best_k = select_k_best(X_train, y_train, estimator, k_values=[2, 5, 7, 10, 12, 13])
    

    with mlflow.start_run():
        
        # Build a training Pipeline
     
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k= best_k)),
            ("model", RandomForestClassifier())])
        
        
        # Hyperparameter Optimization
         
        param_distributions = {
        "model__n_estimators": sp_randint(50, 200),
        "model__max_depth": sp_randint(3, 10),
        "model__min_samples_split": sp_randint(2, 20),
        "model__min_samples_leaf": sp_randint(1, 10),
        }

        
        
        random_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions, n_iter=50,
                                   cv=5, n_jobs=-1, verbose=2)
        
        random_search.fit(X_train, y_train)
        
        # Infer the model signature
        signature = infer_signature(X, random_search.predict(X))
        
        # Model Evaluation
        (cv_score,roc_auc,average_precision,accuracy,precision,recall,f1) = evaluate_model(random_search, X, y, X_test, y_test)

        print("cv_score: %s" % cv_score)
        print("best params: %s" % random_search.best_params_)
        print("roc_auc: %s" % roc_auc)
        print("average_precision: %s" % average_precision)
        print("accuracy: %s" % accuracy)
        print("precision: %s" % precision)
        print("recall: %s" % recall)
        print("f1 score: %s" % f1)

        
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("mean_cv_score", cv_score)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("average_precision", average_precision)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1 score", f1)

        # Register the model           
        mlflow.sklearn.log_model(random_search, "model", signature=signature, registered_model_name="RandomForestModel")

#------------------- Train Gradient Boosting model

if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    
    # Read the final csv file
    csv = (
            "C:/Users/willi/Python/Spotify_Project/Data/final_data.csv"
        )
    try:
            data = pd.read_csv(csv, sep=",", dtype = dtype)
            
    except Exception as e:
            logger.exception(
                "Unable to download training & test CSV. Error: %s", e
            )

    # Split the data into training and test sets. (0.8, 0.2) split.
        
    X = data.drop("mood", axis=1)
    y = data["mood"]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform Feature Selection to find the best K
    
    def select_k_best(X, y, estimator, k_values=[2, 5, 7, 10, 12, 13]):
        best_k = 0
        best_score = float('-inf')
        best_selector = None
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        for k in k_values:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(k=k)),
                ("model", estimator)])
            scores = cross_val_score(pipeline, X, y, cv=cv)
            if scores.mean() > best_score:
                best_k = k
                best_score = scores.mean()
                best_selector = pipeline.named_steps["selector"]
                best_selector.fit(X, y)
                selected_features = X.columns[best_selector.get_support()]
                print(f"Best k: {best_k}")
                print(f"Selected features: {list(selected_features)}")
        return best_k
    
    estimator = GradientBoostingClassifier()
    best_k = select_k_best(X_train, y_train, estimator, k_values=[2, 5, 7, 10, 12, 13])
    

    with mlflow.start_run():
        
        # Build a training Pipeline
     
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k = best_k)),
            ("model", GradientBoostingClassifier())])
        
        
        # Hyperparameter Optimization
         
        param_distributions = {
        "model__n_estimators": sp_randint(50, 200),
        "model__max_depth": sp_randint(3, 10),
        "model__min_samples_split": sp_randint(2, 20),
        "model__min_samples_leaf": sp_randint(1, 10),
        "model__learning_rate": sp_uniform(0.001, 0.1)
        }

        
        
        random_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions, n_iter=50,
                                   cv=5, n_jobs=-1, verbose=2)
        
        random_search.fit(X_train, y_train)
        
        # Infer the model signature
        signature = infer_signature(X, random_search.predict(X))
        
        # Model Evaluation
        (cv_score,roc_auc,average_precision,accuracy,precision,recall,f1) = evaluate_model(random_search, X, y, X_test, y_test)

        print("cv_score: %s" % cv_score)
        print("best params: %s" % random_search.best_params_)
        print("roc_auc: %s" % roc_auc)
        print("average_precision: %s" % average_precision)
        print("accuracy: %s" % accuracy)
        print("precision: %s" % precision)
        print("recall: %s" % recall)
        print("f1 score: %s" % f1)

        
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("mean_cv_score", cv_score)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("average_precision", average_precision)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1 score", f1)

        # Register the model     
        mlflow.sklearn.log_model(random_search, "model", signature=signature, registered_model_name="GradientBoostingModel")
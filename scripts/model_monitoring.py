# Imports
import numpy as np
import pandas as pd
from model_deployment import mood_prediction
from pathlib import Path

from evidently.pipeline.column_mapping import ColumnMapping
from evidently.options import ColorOptions
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import TargetDriftPreset
from evidently.metric_preset import ClassificationPreset

from evidently.metrics import (
    ClassificationQualityMetric,
    ClassificationClassBalance,
    ClassificationConfusionMatrix,
    ClassificationQualityByClass,
    ClassificationQualityByFeatureTable,
    ConflictTargetMetric,
    ConflictPredictionMetric 
)

#Data Types dictionary
dtype = {
       'popularity':'int8',
       'genres': 'int8',
       'sub-genres': 'int8',
       'explicit':'int8', 
       'followers': int, 
       'danceability':float,
       'energy': float, 
       'key':'int8',
       'loudness': float,
       'mode':'int8', 
       'instrumentalness':'int8',
       'liveness':'int8',
       'tempo':float, 
       'duration_ms':int,
       'time_signature':'int8',
       'mood': 'int8' 
       }

#Load Data
csv = (
        "C:/Users/willi/Python/Spotify_Project/Data/preprocess_data.csv"
        )

data = pd.read_csv(csv, sep=",", dtype = dtype)

csv = (
        "C:/Users/willi/Python/Spotify_Project/Data/preprocess_new_data.csv"
        )

new_data = pd.read_csv(csv, sep=",",dtype=dtype)

#Configurations
target = 'mood'
prediction = 'prediction'
numerical_features = ['popularity', 'followers', 'danceability', 'energy', 'loudness', 'tempo','duration_ms']
categorical_features = ['genres', 'sub-genres', 'explicit','liveness','instrumentalness','key','mode','time_signature']

reports_dir = Path('C:/Users/willi/Python/Spotify_Project/reports') #/ f'{today}'
reports_dir.mkdir(exist_ok=True)


# Model training
X_data = data.drop("mood", axis=1)
X_new_data = new_data.drop("mood", axis=1)

data['prediction']= mood_prediction(X_data)
new_data['prediction'] = mood_prediction(X_new_data)

#Type adjustment
data['mood'] = data['mood'].astype('str')
data['prediction'] = data['prediction'].astype('str')

new_data['mood'] = new_data['mood'].astype('str')
new_data['prediction'] = new_data['prediction'].astype('str')


# Model Monitoring
reference_data = data
current_data = new_data

column_mapping = ColumnMapping()

column_mapping.target = target
column_mapping.prediction = prediction
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features
column_mapping.pos_label = '1'

# Model perfomance

#label binary classification
classification_report = Report(metrics=[
    ClassificationQualityMetric(),
    ClassificationClassBalance(),
    ConflictTargetMetric(),
    ConflictPredictionMetric(),
    ClassificationConfusionMatrix(),
    ClassificationQualityByClass(),
    ClassificationQualityByFeatureTable(columns = numerical_features),
])

classification_report.run(reference_data = reference_data, current_data= current_data, column_mapping=column_mapping)
classification_report_path = reports_dir / 'classification_report.html'
classification_report.save_html(classification_report_path)

# Target drift

target_drift_report = Report(metrics=[TargetDriftPreset()])
target_drift_report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)

target_drift_report_path = reports_dir / 'target_drift.html'
target_drift_report.save_html(target_drift_report_path)

# Data drift
column_mapping = ColumnMapping()
column_mapping.numerical_features = numerical_features

data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)
data_drift_report_path = reports_dir / 'data_drift.html'
data_drift_report.save_html(data_drift_report_path)

# Data quality
column_mapping = ColumnMapping()
column_mapping.numerical_features = numerical_features

data_quality_report = Report(metrics=[DataQualityPreset()])
data_quality_report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)

data_quality_report_path = reports_dir / 'data_quality.html'
data_quality_report.save_html(data_quality_report_path)

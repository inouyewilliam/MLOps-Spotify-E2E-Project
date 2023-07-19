# Building an automated MLOps pipeline and recommending an open-source stack to deploy a ML Application

## Introduction
Welcome to the MLOps project! As the field of machine learning operations (MLOps) gains momentum, the need for continuous ML model deployment and efficient management throughout the lifecycle becomes crucial. This project addresses these challenges by building a robust CI/CD pipeline using open source machine learning tools, guided by the CRISP-ML(Q) methodology and Google's maturity model framework.

The goal is to enable smooth deployment of ML applications, ensuring continuous monitoring of model performance, triggering retraining when needed. For this study, it is focused on a supervised learning task involving the Spotify Web API, where tracks are sorted into mood playlists, specifically happy and sad.

To adhere to MLOps standards, the CRISP-ML(Q) methodology is followed, ensuring successful integration of ML systems into production. The selected MLOps stack encompasses powerful tools such as Git, DVC, GitHub Actions and MLflow that facilitate code and data versioning, ensuring reproducible ML models.

The scalability requirement is met by leveraging Spotify Web API and Streamlit. Also, the crucial aspect of monitoring MLOps using reports Evidently evaluates data quality, data skew and model classification performance.

The results show the effectiveness of the pipeline in meeting MLOps requirements, as well as highlighting the explainability and responsible aspects of the AI. This is achieved by employing the Shap library to identify key features in model predictions.

Overall, a comprehensive MLOps project is presented, combining ML industry best practices with an open source MLOps stack. I hope this project serves as a valuable resource for those looking to implement efficient and scalable ML solutions in real-world applications.

## MLOps definition 
![image](https://github.com/inouyewilliam/MLOps-Spotify-E2E-Project/assets/62669400/7899bbfb-5cc1-43dc-b2cb-fdf0b6625e8b)
<h5 align="center"><em>Summary of DevOps and MLOps principles</em></h5>

## Methodologies followed
![image](https://github.com/inouyewilliam/MLOps-Spotify-E2E-Project/assets/62669400/3d101722-bbce-484a-b457-485309636335)
<h5 align="center"><em>Google's Maturity Framework</em></h5>

![image](https://github.com/inouyewilliam/MLOps-Spotify-E2E-Project/assets/62669400/91d0b5f1-6a1d-42dc-84d9-87aa932a257d)
<h5 align="center"><em>CRISP-ML(Q) Methology</em></h5>

## Chosen MLOps stack 
![image](https://github.com/inouyewilliam/MLOps-Spotify-E2E-Project/assets/62669400/bb6d0b08-a3f5-401f-8305-96bc12f6d33d)
<h5 align="center"><em>MLOps stack in Google’s maturity model framework</em></h5>

## Final ML Application
😎 **Deployed Streamlit app: https://music-mood-prediction.streamlit.app/**


## API reference
https://spotipy.readthedocs.io/en/2.22.0/

## My Dagshub reference
https://dagshub.com/inouyewilliam/MLOps-Spotify-E2E-Project

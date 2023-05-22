import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder

df = pd.read_csv("C:/Users/willi/Python/Spotify_Project/Data/raw_data.csv", sep=",")

df_new = pd.read_csv("C:/Users/willi/Python/Spotify_Project/Data/new_data.csv", sep=",")

def preprocess(df):
       # Change order of columns
       df = df.loc[:,['artist', 'album', 'track_name', 'release_date', 'popularity', 'genres','sub-genres',
              'explicit', 'followers', 'track_id', 'danceability', 'energy', 'key',
              'loudness', 'mode', 'speechiness', 'instrumentalness', 'liveness',
              'valence', 'tempo', 'duration_ms', 'time_signature']]

       # Clean Other columns
       df["mood"] = [1 if i >= 0.5 else 0 for i in df.valence]
       df["liveness"] = [1 if i >= 0.8 else 0 for i in df.liveness]
       df["explicit"] = [1 if i == True else 0 for i in df.explicit]
       df["speechiness"] = [1 if i > 0.66 else 0 for i in df.speechiness]
       df["instrumentalness"] = [1 if i > 0.5 else 0 for i in df.instrumentalness]

       # Drop column valence
       df = df.drop(columns = ["valence"],axis = 1)

       # Improve data types
       df['popularity'] = df['popularity'].astype('int8')
       df['time_signature'] = df['time_signature'].astype('int8')
       df['key']  = df['key'].astype('int8')
       df['mood']  = df['mood'].astype('int8')

       df['mode'] = df['mode'].astype('int8')
       df['explicit'] =  df['explicit'].astype('int8')         
       df['instrumentalness'] = df['instrumentalness'].astype('int8')
       df['liveness'] = df['liveness'].astype('int8')

       # Transform nan to zeros and 
       df["genres"].replace(np.nan, 0, inplace=True)
       df["sub-genres"].replace(np.nan, 0, inplace=True)

       # Encode genres and sub-genres
       le = LabelEncoder()
       df['genres'] = le.fit_transform(df['genres'].astype(str))
       df['sub-genres'] = le.fit_transform(df['sub-genres'].astype(str))
       
       # Build train dataset
       df = df.drop(columns = ['artist', 'album', 'track_name','release_date','track_id','speechiness'], axis = 1)
       
       return df


#Use preprocess function      
train = preprocess(df)
new_train = preprocess(df_new)

#Concatenate the dataframes
final_train = pd.concat([train, new_train])

#Save the complete preprocess data
train.to_csv("C:/Users/willi/Python/Spotify_Project/Data/preprocess_data.csv", index = False)

new_train.to_csv("C:/Users/willi/Python/Spotify_Project/Data/preprocess_new_data.csv", index = False)

final_train.to_csv("C:/Users/willi/Python/Spotify_Project/Data/final_data.csv", index = False)
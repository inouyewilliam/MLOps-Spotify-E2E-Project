import streamlit as st
import pandas as pd
import numpy as np
from model_deployment import mood_prediction

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# ----------- Delete Default buttons

#st.markdown("""
#<style>
#.css-nqowgj.edgvbvh3
#{
#visibility:hidden;
#}
#.css-h5rgaw.egzxvld1
#{
#visibility:hidden;
#}
#</style>
#""", unsafe_allow_html = True)


# ----------- General things
st.title('Music Mood Prediction App')
st.divider()

# ----------- Sidebar
st.sidebar.header('Dashboard `Spotify`')
st.sidebar.subheader('User Input Features')

def user_input_features():
        popularity = st.sidebar.slider('popularity', 0, 100, 50)
        genres = st.sidebar.slider('genres', 0, 200, 100)
        sub_genres = st.sidebar.slider('sub_genres', 0, 200, 100)
        explicit  = st.sidebar.selectbox('explicit', ('1','0'))
        followers = st.sidebar.slider('followers', 0, 100000000, 50000000)
        danceability = st.sidebar.slider('danceability', 0.0, 1.0, 0.5)
        energy = st.sidebar.slider('energy', 0.0, 1.0, 0.5)
        key = st.sidebar.slider('key', 0, 11, 5)
        loudness = st.sidebar.slider('loudness', -60, 0, -30)
        mode = st.sidebar.selectbox('mode', ('1','0'))
        tempo = st.sidebar.slider('tempo', 60, 200, 100)
        duration_ms = st.sidebar.slider('duration_ms', 90000, 600000, 300000)
        time_signature = st.sidebar.selectbox('time_signature', ('1','3','4','5'))
        data = {'popularity': popularity,
                 'genres':genres,
                 'sub-genres':sub_genres,
                 'explicit': explicit,
                 'followers' :followers,
                 'danceability' :danceability,
                 'energy' :energy,
                 'key' : key,
                 'loudness' : loudness,
                 'mode' : mode,
                 'tempo' : tempo,
                 'duration_ms' : duration_ms,
                 'time_signature' : time_signature
                    }
        features = pd.DataFrame(data, index=[0])
        return features
    
input_df = user_input_features()

st.sidebar.markdown('''
---
Created 😎 by [William Inouye](https://github.com/inouyewilliam/)
''')

# ----------- Predictor Page

st.subheader("User input features")
st.dataframe(input_df)

st.subheader("File uploader")
file = st.file_uploader("Upload music data", type = ["csv"])

if file is not None:          
    # Open DataFrame
    try:
        temp_filename = "temp.csv"
        with open(temp_filename, "wb") as f:
            f.write(file.getbuffer())
                
        data = pd.read_csv(temp_filename)
        if data.empty:
            st.write("Error: The file is empty.")
        else:
            st.write("Dataframe:")
            st.dataframe(data)
    except pd.errors.ParserError:
            st.write("Error: Invalid CSV file.")
 
                    
    # Make Predictions file          
    predictions_button = st.button("Mood Prediction")
    
    if predictions_button and not data.empty:
        
            predictions, predictions_proba = mood_prediction(temp_filename)
            
            st.subheader("Mood Predictions")
            
            for index, mood in enumerate(predictions):        
                if mood == 1:
                    st.write(f"{index} 😊happy")
                else:
                    st.write(f"{index} 😒sad")
                    
            st.subheader('Prediction Probabilities')
            formatted_probabilities = np.apply_along_axis(lambda x: ['{:.2f}%'.format(i * 100) for i in x], axis=1, arr=predictions_proba)
            df = pd.DataFrame(formatted_probabilities, columns=['0', '1'])
            st.dataframe(df)

else:         
    # Make Predictions user input          
    predictions_button = st.button("Mood Prediction")
    
    if predictions_button and not input_df.empty:
        
            predictions, predictions_proba = mood_prediction(input_df)
            
            st.subheader("Mood Prediction")
            
            for index, mood in enumerate(predictions):        
                if mood == 1:
                    st.write(f"{index} 😊happy")
                else:
                    st.write(f"{index} 😒sad")
                    
            st.subheader('Prediction Probability')
            formatted_probabilities2 = np.apply_along_axis(lambda x: ['{:.2f}%'.format(i * 100) for i in x], axis=1, arr=predictions_proba)
            df2 = pd.DataFrame(formatted_probabilities2, columns=['0', '1'])
            st.dataframe(df2)                   
            

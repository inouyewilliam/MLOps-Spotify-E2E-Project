import streamlit as st
from streamlit_option_menu import option_menu
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

# ----------- options menu
selected3 = option_menu(None, ["Home", "Upload",  "Prediction"], 
    icons=['house', 'cloud-upload', "music-player"], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    }
)

# ----------- Sidebar
st.sidebar.header('Dashboard `Spotify`')

st.sidebar.markdown('''
---
Created with 😎 by [William Inouye](https://github.com/inouyewilliam/)
''')

# ----------- Predictor Page

    
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
                    
    # Make Predictions           
    predictions = st.button("Mood Prediction")
    if predictions and not data.empty:
            moods = mood_prediction(temp_filename)
            st.write("Mood Predictions:")
            for index, mood in enumerate(moods):        
                if mood == 1:
                    st.write(f"{index} 😊happy")
                else:
                    st.write(f"{index} 😒sad")
                        
            

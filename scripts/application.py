import streamlit as st
import pandas as pd
import numpy as np
from model_deployment import mood_prediction

# ----------- Delete Default buttons
st.markdown("""
<style>
.css-nqowgj.edgvbvh3
{
visibility:hidden;
}
.css-h5rgaw.egzxvld1
{
visibility:hidden;
}
</style>
""", unsafe_allow_html = True)

# ----------- General things
st.title('Music Mood Prediction App')
st.divider()


# ----------- Sidebar
page = st.sidebar.selectbox('Page Navigation', ["Predictor", "Music Details"])

# ----------- Predictor Page
if page == "Predictor":
    
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
                for mood in moods:         
                    if mood == 1:
                        st.write("😊happy")
                    else:
                        st.write("😒sad")
                        
            
# ----------- Music Details Page
#if page == "Music Details":
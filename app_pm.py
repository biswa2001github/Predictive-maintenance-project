import numpy as np
import pandas as pd
import streamlit as st
import pickle
import json
import base64
from sklearn.preprocessing import StandardScaler

pickle_model = open('predictive_maintenance_model.pickle', 'rb')
classifier = pickle.load(pickle_model)

with open('columns.json') as f:
    data = json.load(f)
data_columnns = data['data-columns']

def predict_failure(air_temperature, process_temperature, rpm, torque, tool_wear, type):
    try:
        loc_index = data_columnns.index(type.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columnns))
    x[0] = air_temperature
    x[1] = process_temperature
    x[2] = rpm
    x[3] = torque
    x[4] = tool_wear
    
    if loc_index>=0:
        x[loc_index]=1
    X = pd.read_csv('X_data_for_scaling.csv')
    scaler = StandardScaler()
    scaler.fit(X.values)
    z = scaler.transform([x])
    return classifier.predict(z)

def main():
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('Background_image.jpg')

    st.title('Predictive Maintenance of Machines')
    st.write('Welcome! Predictive maintenance is a technique that uses data analysis tools and techniques to detect anomalies in the operation and possible defects in equipment and processes so that it is possible to fix it before the failure.')
    st.write('Please enter the required values of parameters to predict the failure.')
    st.caption('Please ensure to put correct categories of values to avoid error in prediction.')

    type = st.selectbox('Type of Machine',
     data_columnns[5:], key='good', help='Type_h corresponds for High Product quality, Type_m corresponds for medium qulaity and Type_l for low quality')
    air_temperature = st.text_input('Air Temperature [Kelvin]', max_chars=4)
    process_temperature = st.text_input('Process Temperature [Kelvin]', max_chars=4)
    rpm = st.text_input('Rotational Speed [RPM]', max_chars=4)
    torque = st.text_input('Toruqe [Nm]', max_chars=4)
    tool_wear = st.text_input('Tool Wear [min]', max_chars=4)
    result=''

    if st.button('PREDICT'):
        result=predict_failure(air_temperature, process_temperature, rpm, torque, tool_wear, type)
        st.success(f'{result}')

if __name__ == '__main__':
    main()
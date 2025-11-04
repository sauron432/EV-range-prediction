import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
st.set_page_config(
    page_title='EV Range Predictor',
    layout='wide',
    initial_sidebar_state='expanded'
)
pickle_file = 'RF_regressor.pkl'
try:
    with open(pickle_file,'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f'Error:{pickle_file} not found!')
    st.stop()
except Exception as e:
    st.error(f'Error loading model {e}')
    st.stop()

st.title("EV Range Prediction Project")
st.write('----------')

user_input_data = {}
st.sidebar.header('EV specification inputs')
st.sidebar.markdown('Please input the parameters below:')

top_speed = st.sidebar.slider(
    '1. Top speed (kmh)',
    min_value=125,max_value=325,value=150, step=1
)

battery_capacity = st.sidebar.slider(
    '2. Battery Capacity (kWh)',
    min_value=20,max_value=120,value=50, step=1    
)

torque = st.sidebar.slider(
    '3. Torque (nm)',
    min_value=110,max_value=1350,value=450, step=1       
)

acceleration = st.sidebar.slider(
    '4. Acceleration (0-100)',
    min_value=2.2,max_value=20.0,value=5.0, step=0.1       
)

fast_charge_power = st.sidebar.slider(
    '5. Fast Charge power (kW DC)',
    min_value=30,max_value=350,value=80, step=1     
)

user_input_data['top_speed_kmh'] = top_speed
user_input_data['battery_capacity_kWh'] = battery_capacity
user_input_data['torque_nm'] = torque
user_input_data['acceleration_0_100_s'] = acceleration
user_input_data['fast_charging_power_kw_dc'] = fast_charge_power

input_df = pd.DataFrame([user_input_data])
scaler = StandardScaler()
scaled_input = scaler.fit_transform(input_df)

if st.sidebar.button("Predict range", type='primary'):
    prediction = model.predict(scaled_input)[0]
    st.subheader("Predicted Range")
    st.metric(
        label='',
        value = f'{round(prediction,0)} km'
    )
st.write('----------')


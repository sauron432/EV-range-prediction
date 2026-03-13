import streamlit as st
import pandas as pd
import requests
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='EV Range Predictor',
    layout='wide',
    initial_sidebar_state='expanded'
)

API_URL = "http://ev_api:8000/predict"

st.title("EV Range Prediction Project")
st.write('----------')

user_input_data = {}
st.sidebar.header('EV specification inputs')
st.sidebar.markdown('Please input the parameters below:')

top_speed = st.sidebar.slider(
    '1. Top speed (kmh)',
    min_value=125, max_value=325, value=125, step=1
)

battery_capacity = st.sidebar.slider(
    '2. Battery Capacity (kWh)',
    min_value=20, max_value=120, value=20, step=1
)

torque = st.sidebar.slider(
    '3. Torque (nm)',
    min_value=110, max_value=1350, value=110, step=1
)

acceleration = st.sidebar.slider(
    '4. Acceleration (0-100)',
    min_value=2.2, max_value=20.0, value=2.2, step=0.1
)

fast_charge_power = st.sidebar.slider(
    '5. Fast Charge power (kW DC)',
    min_value=30, max_value=350, value=30, step=1
)

user_input_data = {
    'top_speed_kmh': top_speed,
    'battery_capacity_kWh': battery_capacity,
    'torque_nm': torque,
    'acceleration_0_100_s': acceleration,
    'fast_charging_power_kw_dc': fast_charge_power
}

input_df = pd.DataFrame([user_input_data])

try:
    response = requests.post(API_URL, json=user_input_data)
    result = response.json()

    col1, col2 = st.columns([1, 2])

    with col1:
        display_df = input_df.T.rename(
            columns={0: 'Input features'},
            index={
                'top_speed_kmh': 'Top Speed (km/h)',
                'battery_capacity_kWh': 'Battery Capacity (kWh)',
                'torque_nm': 'Torque (Nm)',
                'acceleration_0_100_s': 'Acceleration (0-100 s)',
                'fast_charging_power_kw_dc': 'Fast Charge Power (kW DC)'
            }
        )

    with col2:
        st.subheader('Predicted Range')
        if result["status"] == "success":
            st.metric(
                label='  ',
                value=f'{result["predicted_range_km"]} km'
            )
        else:
            st.error(f'Prediction error: {result["message"]}')

    st.dataframe(display_df, use_container_width=True)
    st.markdown('-----')

except requests.exceptions.ConnectionError:
    st.error("Cannot connect to the API. Make sure the API container is running.")
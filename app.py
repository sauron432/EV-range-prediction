import streamlit as st
import pandas as pd
import pickle
st.set_page_config(
    page_title='EV Range Predictor',
    layout='wide',
    initial_sidebar_state='expanded'
)

# The `@st.cache_resource` decorator in Streamlit is used to cache the resource loading function. This
# means that the function decorated with `@st.cache_resource` will only be executed once and the
# result will be cached for subsequent calls. This can help improve the performance of your Streamlit
# app by avoiding unnecessary resource loading operations every time the app is run.
@st.cache_resource
def load_resources():
    pickle_file = st.secrets["MODEL_PATH"]
    scaler_file = st.secrets["SCALER_PATH"]
    try:
        with open(pickle_file,'rb') as file:
            model = pickle.load(file)
        with open(scaler_file,'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error(f'Error:{pickle_file} not found!')
        st.stop()
    except Exception as e:
        st.error(f'Error loading resources {e}')
        st.stop()

model, scaler = load_resources()

st.title("EV Range Prediction Project")
st.write('----------')

user_input_data = {}
st.sidebar.header('EV specification inputs')
st.sidebar.markdown('Please input the parameters below:')

top_speed = st.sidebar.slider(
    '1. Top speed (kmh)',
    min_value=125,max_value=325,value=125, step=1
)

battery_capacity = st.sidebar.slider(
    '2. Battery Capacity (kWh)',
    min_value=20,max_value=120,value=20, step=1    
)

torque = st.sidebar.slider(
    '3. Torque (nm)',
    min_value=110,max_value=1350,value=110, step=1       
)

acceleration = st.sidebar.slider(
    '4. Acceleration (0-100)',
    min_value=2.2,max_value=20.0,value=2.2, step=0.1       
)

fast_charge_power = st.sidebar.slider(
    '5. Fast Charge power (kW DC)',
    min_value=30,max_value=350,value=30, step=1     
)

user_input_data['top_speed_kmh'] = top_speed
user_input_data['battery_capacity_kWh'] = battery_capacity
user_input_data['torque_nm'] = torque
user_input_data['acceleration_0_100_s'] = acceleration
user_input_data['fast_charging_power_kw_dc'] = fast_charge_power

input_df = pd.DataFrame([user_input_data])
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]

col1, col2 = st.columns([1,2])

with col1:
    display_df = input_df.T.rename(
        columns = {0:'Input features'},
        index={
            'top_speed_kmh': 'Top Speed (km/h)',
            'battery_capacity_kWh': 'Battery Capacity (kWh)',
            'torque_nm': 'Torque (Nm)',
            'acceleration_0_100_s': 'Acceleration (0-100 s)',
            'fast_charging_power_kw_dc': 'Fast Charge Power (kW DC)'
        }
    )
    
with col2:
    st.subheader('Predicted range')
    st.metric(
        label = '',
        value = f'{round(prediction,2)} km'
    )

st.dataframe(display_df, use_container_width=True)
st.markdown('-----')
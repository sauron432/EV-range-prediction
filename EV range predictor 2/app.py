import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
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
            scaler= pickle.load(file)
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

battery_capacity = st.sidebar.slider(
    '1. Battery Capacity (kWh)',
    min_value=20,max_value=160,value=160, step=1    
)

price = st.sidebar.slider(
    '2. Price (USD)',
    min_value=5000,max_value=75000,value=75000, step=1     
)

user_input_data['battery_kwh'] = battery_capacity
user_input_data['price_usd'] = price

input_df = pd.DataFrame([user_input_data])
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]

# if st.sidebar.button("Predict range", type='primary'):
#     prediction = model.predict(scaled_input)[0]
#     st.subheader("Predicted Range")
#     st.metric(
#         label='',
#         value = f'{round(prediction,2)} km'
#     )
# st.write('----------')
# st.subheader('EV Specifications Summary')

col1, col2 = st.columns([1,2])

with col1:
    display_df = input_df.T.rename(
        columns = {0:'Input features'},
        index={'battery_kwh':'Batttery Capacity (kWh)',
               'price_usd':'Price (USD)'         
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
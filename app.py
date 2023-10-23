import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

model = joblib.load('xgb_r2.pkl')


# with open('model.pkl', 'wb') as model_file:
#     pickle.dump(model, model_file)

# UI elements for user input
st.title('House Price Prediction')
st.sidebar.header('Input Features')

# Define input fields for relevant features
first_floor_area = st.sidebar.number_input('1st Floor Area (sq. ft.)', min_value=0)
second_floor_area = st.sidebar.number_input('2nd Floor Area (sq. ft.)', min_value=0)
bedrooms = st.sidebar.number_input('Number of Bedrooms', min_value=0)
bsmt_finished_area = st.sidebar.number_input('Basement Finished Area (sq. ft.)', min_value=0)
garage_area = st.sidebar.number_input('Garage Area (sq. ft.)', min_value=0)
living_area = st.sidebar.number_input('Living Area (sq. ft.)', min_value=0)
lot_area = st.sidebar.number_input('Lot Area (sq. ft.)', min_value=0)
mas_vnr_area = st.sidebar.number_input('Masonry Veneer Area', min_value=0)
open_porch_area = st.sidebar.number_input('Open Porch Area (sq. ft.)', min_value=0)
overall_condition = st.sidebar.slider('Overall Condition', min_value=1, max_value=10, step=1)
overall_quality = st.sidebar.slider('Overall Quality', min_value=1, max_value=10, step=1)
total_bsmt_area = st.sidebar.number_input('Total Basement Area (sq. ft.)', min_value=0)
year_built = st.sidebar.number_input('Year Built', min_value=0)
year_remod_add = st.sidebar.number_input('Year Remodeled', min_value=0)


# Predict button
if st.button('Predict'):
    # Prepare user input as a DataFrame
    user_input = pd.DataFrame({
        '1stFlrSF': [first_floor_area],
        '2ndFlrSF': [second_floor_area],
        'BedroomAbvGr': [bedrooms],
        'BsmtFinSF1': [bsmt_finished_area],
        'GarageArea': [garage_area],
        'GrLivArea': [living_area],
        'LotArea': [lot_area],
        'MasVnrArea': [mas_vnr_area],
        'OpenPorchSF': [open_porch_area],
        'OverallCond': [overall_condition],
        'OverallQual': [overall_quality],
        'TotalBsmtSF': [total_bsmt_area],
        'YearBuilt': [year_built],
        'YearRemodAdd': [year_remod_add]
    })


    user_df = pd.DataFrame(user_input, index=[0])


    # Make prediction using the loaded model
    prediction = model.predict(user_df)

    # Display the prediction
    st.subheader('Prediction')
    st.write(f'The predicted house price is: ${prediction[0]:,.2f}')

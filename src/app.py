import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from src.logger import logging 
from src.utils import load_object
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

st.title('Insurance Expense Prediction')
age=st.text_input('Enter your age')
sex=st.selectbox('Select Your Gender',options=['male','female'])	
weight=st.text_input("Enter your weight")
height=st.text_input("Enter your height")

children=st.text_input("How many children you have")
smoker=st.selectbox("Smoking?",options=['yes','no'])
region=st.selectbox("Select Your Region",options=['southwest', 'southeast', 'northwest', 'northeast'])
button=st.button('Check Your expense')

def predict():
    age=int(age)
    weight=int(weight)
    height=int(height)
    children=int(children)
    bmi=weight/((height/100)**2)
    data=CustomData(
        age=age,
        sex=sex,
        bmi=bmi,
        children=children,
        smoker=smoker,
        region=region

    )
    
    final_new_data=data.get_data_as_dataframe()
    predict_pipeline=PredictPipeline()
    pred=predict_pipeline.predict(final_new_data)
    results=round(pred[0],2)
    return results

if button:
    age=int(age)
    weight=int(weight)
    height=int(height)
    children=int(children)
    bmi=weight/((height/100)**2)
    data=CustomData(
        age=age,
        sex=sex,
        bmi=bmi,
        children=children,
        smoker=smoker,
        region=region

    )
    
    
    final_new_data=data.get_data_as_dataframe()
    print(final_new_data)

    predict_pipeline=PredictPipeline()
    pred=predict_pipeline.predict(final_new_data)
    results=round(pred[0],2)
    
    st.text_area('Your Expense will be:',results)

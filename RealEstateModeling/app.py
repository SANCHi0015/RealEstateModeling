import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Sample Data (You will replace this with your actual dataset)
def loadData():
    with open('df.pkl' , 'rb') as file:
        data = pickle.load(file)
    return data

def loadModel():
    with open('pipeline.pkl' ,'rb') as file:
        model = pickle.load(file)
    return model


df = loadData()
model = loadModel()


# Sidebar for User Input
st.sidebar.header('Sidebar')
st.sidebar.selectbox('Select what to perform',['Predictions'])


st.title("Property Features")

property_type = st.selectbox("Property Type", df['property_type'].unique())
sector = st.selectbox("sector", df['sector'].unique())
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1)
agePossession = st.selectbox("Age of Possession", df['agePossession'].unique())
area = st.number_input("Area in Sq. Ft.", min_value=500, max_value=10000, value=1000)
servant_room = st.selectbox("Servant Room", df['servant room'].unique())
store_room = st.selectbox("Store Room", df['store room'].unique())
furnishing_type = st.selectbox("Furnishing Type", df['furnish'].unique())
luxury_category = st.selectbox("Luxury Category", df['features'].unique())
floor_category = st.selectbox("Floor Category", df['floor'].unique())

button = st.button('Predict')

if button == True:
    # Section 1: Price Prediction
    st.header("Price Prediction")
    
    
    # Predict price based on user input
    user_input = pd.DataFrame({
        'property_type': [property_type],
        'sector': [sector],
        'bedRoom': [bedrooms],
        'bathroom': [bathrooms],
        'balcony': [balcony],
        'agePossession': [agePossession],
        'area': [area],
        'servant room': [servant_room],
        'store room': [store_room],
        'furnish': [furnishing_type],
        'features': [luxury_category],
        'floor': [floor_category]
    })    
    
    predicted_price = model.predict(user_input)
    st.success(f"Price: ₹{predicted_price[0]:2f} Cr")

else:
    st.error('Fill property Features')









    # # Section 2: Recommender System
    # st.header("Property Recommender System")
    # # Filter recommendations based on user input
    # recommendations = df[(df['Property Type'] == property_type) &
    #                      (df['Bedrooms'] == bedrooms) &
    #                      (df['Bathrooms'] == bathrooms) &
    #                      (df['Area'] <= area * 1.1) & (df['Area'] >= area * 0.9)]
    
    # st.write("Recommended Properties Based on Your Preferences")
    # st.dataframe(recommendations)
    
    
    





    # # Section 3: Analytical Section
    # st.header("Analytics Dashboard")
    # st.write("Data Overview")


import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load dataset
df = pd.read_csv("Final_Project.csv")

# Load model
reg = pickle.load(open('regression_model.pkl','rb'))

# Load encoders
le_region = pickle.load(open('le_region.pkl','rb'))
le_property_age = pickle.load(open('le_property_age.pkl','rb'))
le_area_type = pickle.load(open('le_area_type.pkl','rb'))

# ----------- PREDICTION FUNCTION -----------
def predict_price(region, area_sqft, floor_no, bedroom, property_age, bathroom):

    region = le_region.transform([region])[0]
    property_age = le_property_age.transform([property_age])[0]

    area_type = df['Area_Tpye'].mode()[0]
    area_type = le_area_type.transform([area_type])[0]

    rate_sqft = df['Rate_SqFt'].mean()

    x = [[region, property_age, area_type,
          area_sqft, rate_sqft, floor_no, bedroom, bathroom]]

    return reg.predict(x)[0]


# ----------- STREAMLIT UI -----------
def run_ml_app():
    st.subheader('Please enter the required details :')

    Location = st.selectbox('Select the Location', df['Region'].sort_values().unique())
    Area_SqFt = st.slider("Select Total Area in SqFt", 500, int(max(df['Area_SqFt'])), step=100)
    Floor_No = st.selectbox("Enter Floor Number", df['Floor_No'].sort_values().unique())
    Bathroom = st.selectbox("Enter Number of Bathroom", df['Bathroom'].sort_values().unique())  # not used
    Bedroom = st.selectbox("Enter Number of Bedroom", df['Bedroom'].sort_values().unique())
    Property_Age = st.selectbox('Select the Property Age', df['Property_Age'].sort_values().unique())

    result = ""

    if st.button("Calculate Price"):
        result = predict_price(Location, Area_SqFt, Floor_No, Bedroom, Property_Age, Bathroom)
        st.success('Total Price in Crores : {}'.format(round((result/10) - 5, 2)))


if __name__ == '__main__':
    run_ml_app()

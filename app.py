import streamlit as st #to host ml model on web app
import pandas as pd
import numpy as np
import pickle #to load model
from pycaret.regression import * #to import ml model packages

@st.cache_data   # Use caching to load data only once
def load_data():
    df = pd.read_csv('global-data-on-sustainable-energy (1).csv') 
    #drop columns 
    df.drop(['Renewable-electricity-generating-capacity-per-capita','Financial flows to developing countries (US $)',
        'Latitude','Electricity from nuclear (TWh)','Density\\n(P/Km2)','Entity', 'Year',
       'Longitude', 'Renewables (% equivalent primary energy)','Energy intensity level of primary energy (MJ/$2017 PPP GDP)' ],
        axis=1, inplace=True)
    # Drop rows with missing values in the 'target column
    df = df.dropna(subset=['Access to electricity (% of population)'])
    return df  # Return the modified DataFrame

df = load_data()

# Set up the PyCaret environment
setup(data=df, target='Access to electricity (% of population)')  

# Load your trained PyCaret model
@st.cache_resource  # Use caching to load the model only once
def load_model():
    model = pickle.load(open('my_final_pipeline.pkl','rb'))  
    return model

best_model = load_model()

# Streamlit app
st.title("Percentage Energy Access Prediction")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Prediction"))
#create pages for web app
if page == "Home":
    st.image("tower.png")
    st.info("This project application helps you calculate the percentage energy access for a given country.")

if page == "Prediction":
    st.write("## Prediction")
    st.write("Enter the input values for prediction:")
    
    # Input fields for columns
    cleanfuelShare = st.number_input('Access to clean fuels for cooking (%)', 0, 100)
    renewableShare = st.number_input('Renewable energy share (%)', 0, 100)
    fossilFuelElectricity = st.number_input('Electricity from fossil fuels (TWh)', 0, 100000)
    renewablesElectricity = st.number_input('Electricity from renewables (TWh)', 0, 100000)
    lowcarbonElectricity = st.number_input('Low-carbon electricity (%)', 0, 100)
    consumptionPercapita = st.number_input('Energy consumption per capita', 0, 100000)
    co2Emissions = st.number_input('CO2 emissions', 0, 10000)
    gdpGrowth = st.number_input('GDP growth', 0, 10000)
    gdpPercapita = st.number_input('GDP per capita', 0, 10000)
    land = st.number_input('Land area (Km2)', 0, 100000)
    
    if st.button('Predict'):
        # Create a dictionary with input data
        input_data ={
            'Access to clean fuels for cooking': cleanfuelShare,
            'Renewable energy share in the total final energy consumption (%)': renewableShare,
            'Electricity from fossil fuels (TWh)': fossilFuelElectricity,
            'Electricity from renewables (TWh)': renewablesElectricity,
            'Low-carbon electricity (% electricity)': lowcarbonElectricity,
            'Primary energy consumption per capita (kWh/person)': consumptionPercapita,
            'Value_co2_emissions_kt_by_country': co2Emissions,
            'gdp_growth': gdpGrowth,
            'gdp_per_capita': gdpPercapita,
            'Land Area(Km2)': land
        }

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])

        # Make predictions using PyCaret
        prediction = predict_model(best_model, data=input_df)
        # Make predictions using PyCaret without handling NaNs
        # prediction = best_model.predict(input_df)
        
        st.write(f'Predicted Percentage Energy Access: {prediction["Label"][0]}')



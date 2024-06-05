import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load CSV File
solar_data = pd.read_csv("Solar_power_generation.csv")
weather_data = pd.read_csv("Weather_Sensor.csv")
solar_data.drop(labels='PLANT_ID', axis=1, inplace=True)

#Formating date time
solar_data['DATE_TIME'] = pd.to_datetime(solar_data['DATE_TIME'])

weather_data.drop(labels='PLANT_ID', axis=1, inplace=True)

#Formating date and time
weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'])

#Merge
solar = pd.merge(solar_data,weather_data.drop(columns = ['SOURCE_KEY']), on='DATE_TIME')

#Seprate columns
# Adding separate Time and Date columns
solar["DATE"] = pd.to_datetime(solar["DATE_TIME"]).dt.date
solar["TIME"] = pd.to_datetime(solar["DATE_TIME"]).dt.time
solar['DAY'] = pd.to_datetime(solar['DATE_TIME']).dt.day
solar['MONTH'] = pd.to_datetime(solar['DATE_TIME']).dt.month
solar['WEEK'] = pd.to_datetime(solar['DATE_TIME']).dt.isocalendar().week

# Add Hours and Minutes for ML models
solar['HOURS'] = pd.to_datetime(solar['TIME'], format='%H:%M:%S').dt.hour
solar['MINUTES'] = pd.to_datetime(solar['TIME'], format='%H:%M:%S').dt.minute
solar['TOTAL MINUTES PASS'] = solar['MINUTES'] + solar['HOURS'] * 60

# Add Date as string column
solar["DATE_STRING"] = solar["DATE"].astype(str) # add column with date as string
solar["HOURS"] = solar["HOURS"].astype(str)
solar["TIME"] = solar["TIME"].astype(str)

# Converting 'SOURCE_KEY' Categorial to Numerical
encoder = LabelEncoder()
solar['SOURCE_KEY_NUMBER'] = encoder.fit_transform(solar['SOURCE_KEY'])
solar.head()

X = solar[['DAILY_YIELD','TOTAL_YIELD','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION','DC_POWER']]
Y = solar['AC_POWER']



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


model = RandomForestRegressor()

model.fit(X_train, Y_train)

RF_pred = model.predict(X_test)




# Calculate and print the raw score
raw_score_rfr = 100 * model.score(X_test, Y_test)
#print(f'RF Model Raw Score: {raw_score_rfr:4.4f}%')

# Calculate and print the R2 score
y_pred_rfr = model.predict(X_test)
R2_Score_rfr = (r2_score(y_pred_rfr, Y_test) * 100)
#print(f'R2 Score (RF)= {R2_Score_rfr:4.4f}%')

mse = mean_squared_error(Y_test, RF_pred)
#print("MSE:", mse)

rmse = np.sqrt(mse)
#print("RMSE:", rmse)


# Define the prediction function
def predict_solar_power(DAILY_YIELD,TOTAL_YIELD,AMBIENT_TEMPERATURE,MODULE_TEMPERATURE,IRRADIATION,DC_POWER):
    input_data = np.array([[DAILY_YIELD,TOTAL_YIELD,AMBIENT_TEMPERATURE,MODULE_TEMPERATURE,IRRADIATION,DC_POWER]])
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
st.title("Solar Power Prediction")

st.header("Enter the following parameters:")

DAILY_YIELD = st.text_input("Daily Yield")
TOTAL_YIELD = st.text_input("Total Yield")
AMBIENT_TEMPERATURE = st.text_input("Ambient Temperature")
MODULE_TEMPERATURE= st.text_input("Module Temperature")
IRRADIATION = st.text_input("Irradiation")
DC_POWER = st.text_input("DC Power")

if st.button("Predict Solar Power"):
    prediction = predict_solar_power(DAILY_YIELD,TOTAL_YIELD,AMBIENT_TEMPERATURE,MODULE_TEMPERATURE,IRRADIATION,DC_POWER)
    st.success(f"The predicted solar power output is {prediction:.2f} W")

# To run the app, save this file and use the command:
# streamlit run <filename>.py

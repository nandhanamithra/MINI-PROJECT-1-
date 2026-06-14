import pandas as pd
import joblib

#loading the files
model=  joblib.load("milk_random_forest.pkl")
scaler= joblib.load('scaler.pkl')
encoder= joblib.load("grade_encoder.pkl")

#user inputs
pH = float(input("Enter pH: "))
temperature = float(input("Enter Temperature: "))
taste = int(input("Enter Taste (0/1): "))
odor = int(input("Enter Odor (0/1): "))
fat = int(input("Enter Fat (0/1): "))
turbidity = int(input("Enter Turbidity (0/1): "))
colour = int(input("Enter Colour (240 to 255): "))

#sample imput
sample = pd.DataFrame([{
    "pH": pH,
    "Temprature": temperature,
    "Taste": taste,
    "Odor": odor,
    "Fat ": fat,     
    "Turbidity": turbidity,
    "Colour": colour
}])

#transfrm
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)
grade = encoder.inverse_transform(prediction)

print("Predicted Grade:", grade[0] , "quality milk.")
import pickle
import numpy as np

# Load model
model = pickle.load(open('model/model.pkl', 'rb'))

# Example input
area = float(input("Enter area: "))
bedrooms = int(input("Enter bedrooms: "))

# Dummy location encoding (example)
location = input("Enter location (Chennai/Bangalore/Hyderabad): ")

loc_chennai = 1 if location == "Chennai" else 0
loc_bangalore = 1 if location == "Bangalore" else 0
loc_hyderabad = 1 if location == "Hyderabad" else 0

features = np.array([[area, bedrooms, loc_bangalore, loc_chennai, loc_hyderabad]])

prediction = model.predict(features)

print(f"Predicted Price: ₹{prediction[0]:,.2f}")

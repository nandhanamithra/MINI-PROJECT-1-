import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv(r"C:\flutter_projects\INTERNSHIP\lvl 2\miniproject\milknew.csv")
data = data.drop_duplicates()

#encoders
encoder = LabelEncoder()
data["Grade"] = encoder.fit_transform(data["Grade"])

#data assigning
x = data.drop("Grade", axis=1)
y = data["Grade"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#model training
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)

#prediction and accuracy
y_predict=model.predict(x_test)
accuracy=accuracy_score(y_test,y_predict)
print("ACCURACY",accuracy*100,"%")

#dumping to joblib
joblib.dump(model, "milk_random_forest.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "grade_encoder.pkl")

print("Model, encoders and scalers saved successfully.")

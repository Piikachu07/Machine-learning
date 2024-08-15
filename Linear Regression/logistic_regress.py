import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
data = pd.read_csv('framingham.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values and handle them (e.g., drop or fill)
data.dropna(inplace=True)

# Separate the input variables (X) and the target variable (y)
X = data.iloc[:, :-1].values  # Assuming the target variable is in the last column
y = data.iloc[:, -1].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize the input variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Example: Making predictions on new data
new_data = np.array([[male, age, education,	currentSmoker, cigsPerDay,	BPMeds,	prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate,	glucose]])  # Replace with actual values
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print(f"Prediction for new data: {prediction}")

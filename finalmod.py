import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import streamlit as st

# Function to load data from CSV file
def load_data(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: File '{csv_file_path}' not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to preprocess data and train model
def preprocess_data(df):
    if df is None:
        return None

    # Encoding categorical columns
    encode = OrdinalEncoder()
    df["gender"] = encode.fit_transform(df[["gender"]])
    df["smoking_history"] = encode.fit_transform(df[["smoking_history"]])

    # Handling duplicates
    df.drop_duplicates(inplace=True)

    # Splitting data into train and test sets
    y = df["diabetes"]
    x = df.drop("diabetes", axis=1)
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.15, random_state=42)

    # Training a Decision Tree model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train, y_train)

    return model

# Function to predict diabetes using the trained model
def predict_diabetes(model, input_data):
    try:
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    # Set title 
    st.title("Diabetes Prediction App")

    # Page description
    st.write("Enter patient details to predict diabetes.")

    # Path to CSV file
    csv_file_path = 'Update with the actual CSV file path'
    # Load data
    df = load_data(csv_file_path)
    if df is None:
        return

    # Preprocess data and train model
    model = preprocess_data(df)
    if model is None:
        st.error("Failed to train the model. Please check your data.")
        return

    # Collect patient details for prediction
    st.subheader("Enter Patient Details")
    name = st.text_input("Patient Name", "")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    smoking_history = st.radio("Smoking History", ["Never Smoked", "Ex-smoker", "Current Smoker"])
    BMI = st.slider("BMI", 10.0, 50.0, 25.0)
    glucose = st.slider("Glucose Level", 50, 300, 100)
    hyper_tension = st.selectbox("Hypertension", ["Yes", "No"])
    heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])


    # Encode selected features
    gender_val = 1 if gender == "Male" else 0
    smoking_val = ["Never Smoked", "Ex-smoker", "Current Smoker"].index(smoking_history)
    hyper_tension_val = 1 if hyper_tension == "Yes" else 0
    heart_disease_val = 1 if heart_disease == "Yes" else 0

    # Prepare input data for prediction
    input_data = np.array([[gender_val, age, smoking_val, hyper_tension_val, heart_disease_val, BMI, 5.0, glucose]])
    # Predict diabetes on button click
    if st.button('Predict'):
        prediction = predict_diabetes(model, input_data)
        if prediction is not None:
            if prediction[0] == 1:
                st.error('❗ The patient is predicted to have diabetes.')
            else:
                st.success('✅ The patient is predicted to not have diabetes.')

if __name__ == "__main__":
    main()

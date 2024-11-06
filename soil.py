import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Title for the web app
st.title("Data Upload and Model Training App")

# Upload data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Function to display confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)

# Process the uploaded file
if uploaded_file:
    # Load the data
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(data.head())

    # Select features and target
    if st.button("Train Model"):
        target = st.selectbox("Select the target column", data.columns)
        features = [col for col in data.columns if col != target]

        X = data[features]
        y = data[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train the model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Display results
        st.write(f"Model Accuracy: {accuracy:.2f}")
        st.write("Confusion Matrix:")
        plot_confusion_matrix(cm)
        st.write("Classification Report:")
        st.json(report)

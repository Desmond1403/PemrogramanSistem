# Import necessary libraries
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load the dataset
df = pd.read_csv('/workspaces/PemrogramanSistem/All Samples.csv')

# Clean column names by stripping any extra whitespace
df.columns = df.columns.str.strip()

# Select subset of features and the target label
selected_features = ['Df_NH3', 'volt_NH3', 'MQ_136', 'Vmq136', 'mq136_Ratio', 'MQ_135']
X = df[selected_features]  # Features for the model
y = df['Sampe/Class']      # Target column

# Convert target labels to numeric encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=0)

# Train the RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy with selected features: {accuracy}")

# Save the model and label encoder to files
with open("classifier_selected_features.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)

with open("label_encoder.pkl", "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file)

# Streamlit App
# Load the model and label encoder
with open("classifier_selected_features.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Define the prediction function using the loaded model
def prediction(features):
    pred = classifier.predict([features])
    return pred

# Map the prediction to the original label
def map_prediction_to_class(prediction):
    return label_encoder.inverse_transform([prediction])[0]

# Streamlit App UI
def main():
    st.title("Gas Sensor Array Data Classification")

    # Page design
    html_temp = """
    <div style="background-color:yellow;padding:13px">
    <h1 style="color:black;text-align:center;">Gas Sensor Classification App</h1>
    <h2 style="color:blue;text-align:center;">Benvenuto Desmond Marcellino Letsoin 22220036</h4>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Input fields for features
    df_nh3 = st.number_input("Df_NH3", value=0.0)
    volt_nh3 = st.number_input("volt_NH3", value=0.0)
    mq_136 = st.number_input("MQ_136", value=0.0)
    vmq136 = st.number_input("Vmq136", value=0.0)
    mq136_ratio = st.number_input("mq136_Ratio", value=0.0)
    mq_135 = st.number_input("MQ_135", value=0.0)

    # Collect all input features into a list for prediction
    features = [df_nh3, volt_nh3, mq_136, vmq136, mq136_ratio, mq_135]

    result = ""

    # Prediction button
    if st.button("Predict"):
        # Perform prediction
        prediction_result = prediction(features)

        # Convert the prediction to the original class label
        result = map_prediction_to_class(prediction_result[0])

    st.success(f'The system is classified as: {result}')

# Run the main function
if __name__ == "__main__":
    main()

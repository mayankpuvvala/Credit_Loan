import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Load the trained model
with open('credit_model.pkl', 'rb') as file:
    model = joblib.load(file)

# Page title
st.title("Credit Risk Evaluation")

# Define the tabs
tabs = st.tabs(["Prediction", "Visualizations", "Metrics"])

# ----------------- TAB 1: Prediction ----------------- #
with tabs[0]:
    st.header("Predict Credit Risk")

    # User inputs
    age = st.number_input("Age", min_value=18, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    housing = st.selectbox("Housing", ["Own", "Rent", "Free"])
    saving_accounts = st.selectbox("Saving Accounts", ["Little", "Moderate", "Rich", "Quite Rich"])
    checking_account = st.selectbox("Checking Account", ["Little", "Moderate", "Rich"])
    credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
    duration = st.number_input("Duration (Months)", min_value=1, value=12)
    purpose = st.selectbox("Purpose", ["Car", "Furniture", "Radio/TV", "Education", "Business", "Others"])

    # Prepare input for prediction
    input_data = pd.DataFrame({
        "Age": [age],
        "sex": [1 if sex == "Male" else 0],
        "Housing": [housing],
        "Saving accounts": [saving_accounts],
        "Checking account": [checking_account],
        "Credit amount": [credit_amount],
        "Duration": [duration],
        "Purpose": [purpose]
    })

    # Encode categorical features
    input_data['Housing'] = input_data['Housing'].map({"Own": 0, "Rent": 1, "Free": 2})
    input_data['Saving accounts'] = input_data['Saving accounts'].map({"Little": 0, "Moderate": 1, "Rich": 2, "Quite Rich": 3})
    input_data['Checking account'] = input_data['Checking account'].map({"Little": 0, "Moderate": 1, "Rich": 2})
    input_data['Purpose'] = input_data['Purpose'].map({"Car": 0, "Furniture": 1, "Radio/TV": 2, "Education": 3, "Business": 4, "Others": 5})

    # Fill in missing features
    all_features = [
        "Age", "sex", "Housing", "Saving accounts", "Checking account", "Credit amount", "Duration", "Purpose",
        "Feature_9", "Feature_10", "Feature_11", "Feature_12", "Feature_13", "Feature_14", "Feature_15", "Feature_16",
        "Feature_17", "Feature_18", "Feature_19", "Feature_20", "Feature_21", "Feature_22"
    ]
    for feature in all_features:
        if feature not in input_data.columns:
            input_data[feature] = 0

    input_data = input_data[all_features]

    # Prediction button
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        st.success(f"The predicted credit risk is: {'Good' if prediction == 0 else 'Bad'}")

# ----------------- TAB 2: Visualizations ----------------- #
with tabs[1]:
    st.header("Visualizations")

    # Automatically load available graph images from the "visualizations" folder
    graph_folder = "visualizations"
    graph_files = [f for f in os.listdir(graph_folder) if f.endswith(".png")]

    # Clean up names for display
    graph_display_names = [f.replace("_", " ").replace("-", " ").replace(".png", "") for f in graph_files]
    graph_map = dict(zip(graph_display_names, graph_files))

    selected_graph_name = st.selectbox("Select a Graph", graph_display_names)

    # Show the selected graph
    selected_graph_path = os.path.join(graph_folder, graph_map[selected_graph_name])
    image = Image.open(selected_graph_path)
    st.image(image, caption=selected_graph_name, use_column_width=True)

# ----------------- TAB 3: Metrics ----------------- #
with tabs[2]:
    st.header("Model Performance Metrics")

    # Precomputed metrics
    metrics = {
        "accuracy": 0.70,
        "confusion_matrix": [[160, 18], [57, 15]],
        "fbeta_score_beta_2": 0.2336,
        "classification_report": {
            "class_0": {"precision": 0.74, "recall": 0.90, "f1-score": 0.81, "support": 178},
            "class_1": {"precision": 0.45, "recall": 0.21, "f1-score": 0.29, "support": 72},
            "accuracy": 0.70,
            "macro_avg": {"precision": 0.60, "recall": 0.55, "f1-score": 0.55, "support": 250},
            "weighted_avg": {"precision": 0.66, "recall": 0.70, "f1-score": 0.66, "support": 250}
        },
        "precision_weighted": 0.6559,
        "recall_weighted": 0.70,
        "f1_score_weighted": 0.6591
    }

    st.subheader("Accuracy")
    st.write(f"**Accuracy:** {metrics['accuracy']}")

    st.subheader("Confusion Matrix")
    st.write(pd.DataFrame(metrics["confusion_matrix"], columns=["Predicted Good", "Predicted Bad"], index=["Actual Good", "Actual Bad"]))

    st.subheader("F-beta Score (beta=2)")
    st.write(metrics["fbeta_score_beta_2"])

    st.subheader("Classification Report")
    for label, scores in metrics["classification_report"].items():
        if isinstance(scores, dict):
            st.markdown(f"**{label}**")
            st.write(pd.DataFrame([scores]).T.rename(columns={0: label}))

    st.subheader("Weighted Averages")
    st.write(f"**Precision:** {metrics['precision_weighted']}")
    st.write(f"**Recall:** {metrics['recall_weighted']}")
    st.write(f"**F1-Score:** {metrics['f1_score_weighted']}")

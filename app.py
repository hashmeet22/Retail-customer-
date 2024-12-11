import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

# Load the dataset
st.title("Retail Customer Prediction")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='unicode_escape')

    # Data Overview
    st.header("Data Overview")
    st.write("First 10 rows of the dataset:")
    st.write(df.head(10))

    # Preprocessing and Feature Selection
    st.header("Preprocessing")
    st.write("Preparing the data...")

    # Ensure all columns are numeric or encoded properly
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Define features and target
    st.write("Select Features and Target Column")
    features = st.multiselect("Select Features", df.columns.tolist(), default=df.columns.tolist()[:-1])
    target = st.selectbox("Select Target Column", df.columns.tolist(), index=len(df.columns) - 1)

    if not features or not target:
        st.warning("Please select at least one feature and a target column.")
    else:
        X = df[features]
        y = df[target]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model Selection
        st.header("Model Selection")
        model_type = st.selectbox("Choose a Model", ("Random Forest", "KMeans Clustering"))

        if model_type == "Random Forest":
            # Train Random Forest
            st.write("Training Random Forest...")
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Display results
            st.subheader("Model Performance")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"R² Score: {r2:.2f}")

        elif model_type == "KMeans Clustering":
            # Train KMeans
            st.write("Training KMeans Clustering...")
            n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_train)

            # Assign clusters to data
            clusters = kmeans.predict(X_test)
            df['Cluster'] = kmeans.labels_

            # Display clustering results
            st.subheader("Cluster Analysis")
            st.write("Clustered Data Sample:")
            st.write(df.head(10))

            # Visualize clusters
            st.subheader("Cluster Visualization")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=clusters, palette="viridis")
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            st.pyplot(plt)

        # Allow prediction for user inputs
        st.header("Predict for New Data")
        user_input = {}
        for col in features:
            user_input[col] = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))
        user_input_df = pd.DataFrame([user_input])

        # Scale input and predict
        user_input_scaled = scaler.transform(user_input_df)
        if model_type == "Random Forest":
            user_prediction = model.predict(user_input_scaled)
            st.write(f"Predicted Value: {user_prediction[0]:.2f}")
        elif model_type == "KMeans Clustering":
            user_cluster = kmeans.predict(user_input_scaled)
            st.write(f"Predicted Cluster: {user_cluster[0]}")

else:
    st.write("Please upload a dataset to proceed.")

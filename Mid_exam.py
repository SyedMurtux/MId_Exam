# Streamlit: Combined Code for Phase 1 and Phase 2
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="🚗 Automobile Educational Dashboard",
    page_icon="📊",
    layout="wide",
)

# Load Dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(url)

# File Upload Option
def load_user_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return load_data()

# Sidebar for Dataset Selection
st.sidebar.title("🔍 Explore Options")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV):", type="csv")
df = load_user_data(uploaded_file)

menu = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 Overview",
        "📊 Statistical Insights",
        "📈 Scatter & Regression",
        "📊 Filtered Data",
        "📋 Correlation Matrix",
    ],
)

# Section 1: Overview
if menu == "🏠 Overview":
    st.title("🚗 Automobile Data Analysis for Education")
    st.markdown(
        """
        ### Welcome to the Automobile Data Dashboard!
        - Analyze predefined or uploaded datasets.
        - Interact with dynamic filters and custom charts.
        - Gain meaningful insights with educational context.
        """
    )

    st.header("📂 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.write("### Dataset Summary")
    st.write(df.describe(include='all').T)

# Section 2: Statistical Insights
elif menu == "📊 Statistical Insights":
    st.header("📊 Statistical Insights")

    st.markdown("### Select a Feature for Analysis")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    selected_feature = st.selectbox("Choose a numerical feature:", numerical_columns)

    # Handle NaN values gracefully
    if df[selected_feature].isnull().sum() > 0:
        st.warning(f"The selected feature '{selected_feature}' contains missing values. These will be ignored in calculations.")

    # Basic Statistics
    st.write(f"#### Basic Statistics for {selected_feature}")
    stats = df[selected_feature].dropna().describe()
    st.write(stats)

    # Distribution Plot
    st.write(f"#### Distribution of {selected_feature}")
    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature].dropna(), kde=True, ax=ax, color="#2ecc71")
    st.pyplot(fig)

# Section 3: Scatter & Regression
elif menu == "📈 Scatter & Regression":
    st.header("📈 Scatter & Regression Analysis")

    st.markdown("### Select Features for Analysis")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    x_axis = st.selectbox("Choose the X-axis (numerical):", numerical_columns)
    y_axis = st.selectbox("Choose the Y-axis (numerical):", numerical_columns)

    # Filter Data
    st.markdown("### Apply Filters")
    min_val, max_val = st.slider(
        f"Filter data by {x_axis}:",
        float(df[x_axis].min()), float(df[x_axis].max()),
        (float(df[x_axis].min()), float(df[x_axis].max())),
    )
    filtered_data = df[(df[x_axis] >= min_val) & (df[x_axis] <= max_val)]

    # Scatter Plot
    st.write(f"#### Scatter Plot: {x_axis} vs. {y_axis}")
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_axis, y=y_axis, data=filtered_data, ax=ax, color="#3498db")
    st.pyplot(fig)

    # Regression Plot
    st.write(f"#### Regression Plot: {x_axis} vs. {y_axis}")
    fig, ax = plt.subplots()
    sns.regplot(x=x_axis, y=y_axis, data=filtered_data, ax=ax, color="#e74c3c")
    st.pyplot(fig)

# Section 4: Filtered Data
elif menu == "📊 Filtered Data":
    st.title("📊 Filtered Insights")

    st.markdown("### Apply Filters to Narrow Down Data")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Slider for Price Range
    if 'price' in df.columns:
        min_price, max_price = st.slider(
            "Select Price Range:",
            int(df['price'].min()), int(df['price'].max()),
            (int(df['price'].min()), int(df['price'].max())),
        )
        filtered_df = df[(df['price'] >= min_price) & (df['price'] <= max_price)]
        st.write(f"#### Cars in the price range: ${min_price:,} - ${max_price:,}")
        st.dataframe(filtered_df)

    # Filter for Numerical Columns
    selected_column = st.selectbox("Select a numerical feature to filter:", numerical_columns)
    min_val, max_val = st.slider(
        f"Select range for {selected_column}:",
        float(df[selected_column].min()), float(df[selected_column].max()),
        (float(df[selected_column].min()), float(df[selected_column].max())),
    )
    filtered_df = df[(df[selected_column] >= min_val) & (df[selected_column] <= max_val)]

    st.write(f"### Filtered Data by {selected_column} Range:")
    st.dataframe(filtered_df)

# Section 5: Correlation Matrix
elif menu == "📋 Correlation Matrix":
    st.header("📋 Correlation Matrix")

    st.write("### Correlation Matrix for Numerical Variables")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_data = df[numerical_columns].dropna()

    correlation_matrix = correlation_data.corr()

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown(
        """
        ### Insights:
        - **Positive Correlation**: Two variables increase together.
        - **Negative Correlation**: One variable increases, the other decreases.
        - **No Correlation**: No clear relationship.
        """
    )

# Streamlit: Phase 1 of Ultimate Educational App
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="🚗 Educational Automobile Dashboard",
    page_icon="📊",
    layout="wide",
)

# Load Dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(url)

df = load_data()

# Header
st.title("🚗 Automobile Data Analysis for Education")
st.markdown("#### Explore car features and prices with interactive visualizations and statistical insights!")

# Sidebar for Navigation
st.sidebar.title("🔍 Explore Sections")
menu = st.sidebar.radio(
    "Choose a Section",
    [
        "🏠 Overview",
        "📊 Statistical Insights",
        "📈 Scatter Plots & Regression",
        "📋 Correlation Matrix",
    ],
)

# Section 1: Overview
if menu == "🏠 Overview":
    st.header("🏠 Dataset Overview")
    st.write("### Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.write("### Dataset Description")
    st.markdown(
        """
        This dataset contains attributes of cars, such as:
        - **Engine Size**: Power of the vehicle.
        - **Body Style**: Design and category of the car.
        - **Price**: Cost of the vehicle.
        
        ### Why Learn from This Data?
        - Explore relationships between car specifications and price.
        - Gain practical experience with statistical analysis and visualization.
        """
    )

# Section 2: Statistical Insights
elif menu == "📊 Statistical Insights":
    st.header("📊 Statistical Insights")

    st.markdown("### Select a Feature for Analysis")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    selected_feature = st.selectbox("Choose a numerical feature:", numerical_columns)

    # Basic Statistics
    st.write(f"#### Basic Statistics for {selected_feature}")
    stats = df[selected_feature].describe()
    st.write(stats)

    # Distribution Plot
    st.write(f"#### Distribution of {selected_feature}")
    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, ax=ax, color="#2ecc71")
    st.pyplot(fig)

# Section 3: Scatter Plots & Regression
elif menu == "📈 Scatter Plots & Regression":
    st.header("📈 Scatter Plots & Regression")

    st.markdown("### Select Features for Regression Analysis")
    x_axis = st.selectbox("Choose the X-axis (numerical):", numerical_columns)
    y_axis = st.selectbox("Choose the Y-axis (numerical):", numerical_columns)

    # Scatter Plot
    st.write(f"#### Scatter Plot: {x_axis} vs. {y_axis}")
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_axis, y=y_axis, data=df, ax=ax, color="#3498db")
    st.pyplot(fig)

    # Regression Plot
    st.write(f"#### Regression Plot: {x_axis} vs. {y_axis}")
    fig, ax = plt.subplots()
    sns.regplot(x=x_axis, y=y_axis, data=df, ax=ax, color="#e74c3c")
    st.pyplot(fig)

# Section 4: Correlation Matrix
elif menu == "📋 Correlation Matrix":
    st.header("📋 Correlation Matrix")

    st.write("### Correlation Matrix for Numerical Variables")
    correlation_matrix = df[numerical_columns].corr()

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown(
        """
        ### What is a Correlation Matrix?
        - **Positive Correlation**: As one variable increases, so does the other.
        - **Negative Correlation**: As one variable increases, the other decreases.
        - **No Correlation**: No linear relationship exists.
        
        Use this matrix to identify relationships between variables, such as how engine size correlates with price.
        """
    )

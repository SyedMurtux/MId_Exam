# Streamlit: Combined Code for Phases 1, 2, 3, and 4
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="🚗 Automobile Educational Dashboard",
    page_icon="📊",
    layout="wide",
)

# Custom CSS for Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        color: #34495e;
        margin-bottom: 20px;
    }
    .sub-header {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
    }
    .highlight {
        background-color: #e8f6ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #3498db;
        color: #2c3e50;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-header">🚗 Automobile Data Dashboard</div>', unsafe_allow_html=True)

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

# Sidebar for Dataset Selection and Theme Toggle
st.sidebar.title("🔍 Explore Options")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV):", type="csv")
df = load_user_data(uploaded_file)

# Sidebar: Dark/Light Mode
theme_mode = st.sidebar.radio("Theme Mode:", ["🌞 Light Mode", "🌙 Dark Mode"])
if theme_mode == "🌙 Dark Mode":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #2c3e50;
            color: #ecf0f1;
        }
        .main-header {
            color: #ecf0f1;
        }
        .sub-header {
            color: #ecf0f1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Navigation Menu
menu = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 Overview",
        "📊 Statistical Insights",
        "📈 Scatter & Regression",
        "📊 Filtered Data",
        "📈 3D Scatter Plot",
        "📋 Correlation Matrix & Insights",
    ],
)

# Section 1: Overview
if menu == "🏠 Overview":
    st.title("🏠 Overview")
    st.markdown(
        """
        ### Welcome to the Automobile Data Dashboard!
        - Explore dynamic insights and correlations in automobile datasets.
        - Use our preloaded dataset or upload your own for a custom experience.
        """
    )
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

# Section 4: Filtered Data
elif menu == "📊 Filtered Data":
    st.title("📊 Filtered Insights")

    st.markdown("### Apply Filters to Narrow Down Data")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

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

# Section 5: 3D Scatter Plot
elif menu == "📈 3D Scatter Plot":
    st.header("📈 3D Scatter Plot")

    st.markdown("### Select Features for 3D Visualization")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    x_axis = st.selectbox("Choose the X-axis (numerical):", numerical_columns, index=0)
    y_axis = st.selectbox("Choose the Y-axis (numerical):", numerical_columns, index=1)
    z_axis = st.selectbox("Choose the Z-axis (numerical):", numerical_columns, index=2)

    st.write(f"#### 3D Scatter Plot: {x_axis}, {y_axis}, and {z_axis}")
    fig = px.scatter_3d(
        df,
        x=x_axis,
        y=y_axis,
        z=z_axis,
        color="price" if "price" in df.columns else None,
        title=f"3D Scatter Plot of {x_axis}, {y_axis}, and {z_axis}",
        labels={x_axis: x_axis, y_axis: y_axis, z_axis: z_axis},
    )
    st.plotly_chart(fig, use_container_width=True)

# Section 6: Correlation Matrix & Insights
elif menu == "📋 Correlation Matrix & Insights":
    st.header("📋 Correlation Matrix & Dynamic Insights")

    st.write("### Correlation Matrix for Numerical Variables")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_data = df[numerical_columns].dropna()

    correlation_matrix = correlation_data.corr()

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Dynamic Insights
    st.markdown("### Dynamic Insights")
    highest_corr = correlation_matrix.unstack().sort_values(ascending=False)
    highest_corr = highest_corr[highest_corr < 1].idxmax()  # Exclude self-correlation
    st.write(
        f"The strongest correlation is between `{highest_corr[0]}` and `{highest_corr[1]}` "
        f"with a correlation coefficient of **{correlation_matrix.loc[highest_corr[0], highest_corr[1]]:.2f}**."
    )

# Streamlit-based Python script for the ultimate automobile data analysis dashboard
# Streamlit: Phase 1 of Ultimate Educational App
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Page Configuration
st.set_page_config(
    page_title="🚗 Auto Insights | The Ultimate Dashboard",
    page_icon="✨",
    page_title="🚗 Educational Automobile Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Styling
st.markdown(
    """
    <style>
    /* General App Styling */
    .stApp {
        background-color: #fdfdfd;
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Main Header */
    .main-header {
        font-size: 60px;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 30px;
    }
    /* Section Header */
    .section-header {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #34495e;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    /* Highlights */
    .highlight {
        background-color: #e8f6ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #3498db;
        margin-bottom: 20px;
        font-size: 18px;
        color: #2c3e50;
    }
    /* Footer */
    .footer {
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        color: #7f8c8d;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# App Header
st.markdown('<div class="main-header">🚗 Auto Insights | The Ultimate Dashboard</div>', unsafe_allow_html=True)
st.markdown("#### Bringing automotive data analysis to life with elegance and precision. 🌟")
# Load Data
# Load Dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(url)

df = load_data()

# Sidebar Navigation
# Header
st.title("🚗 Automobile Data Analysis for Education")
st.markdown("#### Explore car features and prices with interactive visualizations and statistical insights!")
# Sidebar for Navigation
st.sidebar.title("🔍 Explore Sections")
menu = st.sidebar.radio(
    "Navigate to:",
    "Choose a Section",
    [
        "🏠 Overview",
        "📈 Regression Insights",
        "📊 Box Plot Analysis",
        "🎨 Interactive Heatmaps",
        "🔑 Key Findings",
        "📊 Statistical Insights",
        "📈 Scatter Plots & Regression",
        "📋 Correlation Matrix",
    ],
)

# Section 1: Overview
if menu == "🏠 Overview":
    st.markdown('<h2 class="section-header">🏠 Dataset Overview</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            '<div class="highlight">This dataset provides a detailed overview of car features and prices, enabling actionable insights.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.write("### Dataset Summary")
        st.write(df.describe(include='all').T)
    st.markdown(
        '<div class="highlight">Features include **numerical** variables like engine size and price, alongside **categorical** variables such as body style and drive wheels.</div>',
        unsafe_allow_html=True,
    )
# Section 2: Regression Insights
elif menu == "📈 Regression Insights":
    st.markdown('<h2 class="section-header">📈 Regression Insights</h2>', unsafe_allow_html=True)
    st.header("🏠 Dataset Overview")
    st.write("### Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.write("### Dataset Description")
    st.markdown(
        '<div class="highlight">Explore linear relationships between car attributes and price using regression plots.</div>',
        unsafe_allow_html=True,
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

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Engine Size vs. Price")
        fig, ax = plt.subplots()
        sns.regplot(x="engine-size", y="price", data=df, ax=ax, color="#2980b9")
        st.pyplot(fig)
    with col2:
        st.write("#### Highway MPG vs. Price")
        fig, ax = plt.subplots()
        sns.regplot(x="highway-mpg", y="price", data=df, ax=ax, color="#e74c3c")
        st.pyplot(fig)
    col3, col4 = st.columns(2)
    with col3:
        st.write("#### Peak RPM vs. Price")
        fig, ax = plt.subplots()
        sns.regplot(x="peak-rpm", y="price", data=df, ax=ax, color="#27ae60")
        st.pyplot(fig)
    with col4:
        st.write("#### Stroke vs. Price")
        fig, ax = plt.subplots()
        sns.regplot(x="stroke", y="price", data=df, ax=ax, color="#8e44ad")
        st.pyplot(fig)
# Section 3: Box Plot Analysis
elif menu == "📊 Box Plot Analysis":
    st.markdown('<h2 class="section-header">📊 Box Plot Analysis</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div class="highlight">Analyze price distributions across categorical features such as body style, engine location, and drive wheels.</div>',
        unsafe_allow_html=True,
    )
# Section 2: Statistical Insights
elif menu == "📊 Statistical Insights":
    st.header("📊 Statistical Insights")

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Price by Body Style")
        fig, ax = plt.subplots()
        sns.boxplot(x="body-style", y="price", data=df, ax=ax, palette="coolwarm")
        st.pyplot(fig)
    st.markdown("### Select a Feature for Analysis")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    selected_feature = st.selectbox("Choose a numerical feature:", numerical_columns)

    with col2:
        st.write("#### Price by Engine Location")
        fig, ax = plt.subplots()
        sns.boxplot(x="engine-location", y="price", data=df, ax=ax, palette="Set3")
        st.pyplot(fig)
    # Basic Statistics
    st.write(f"#### Basic Statistics for {selected_feature}")
    stats = df[selected_feature].describe()
    st.write(stats)

    st.write("#### Price by Drive Wheels")
    # Distribution Plot
    st.write(f"#### Distribution of {selected_feature}")
    fig, ax = plt.subplots()
    sns.boxplot(x="drive-wheels", y="price", data=df, ax=ax, palette="Set2")
    sns.histplot(df[selected_feature], kde=True, ax=ax, color="#2ecc71")
    st.pyplot(fig)

# Section 4: Heatmaps
elif menu == "🎨 Interactive Heatmaps":
    st.markdown('<h2 class="section-header">🎨 Interactive Heatmaps</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div class="highlight">Visualize price variations across categorical features using heatmaps.</div>',
        unsafe_allow_html=True,
    )
# Section 3: Scatter Plots & Regression
elif menu == "📈 Scatter Plots & Regression":
    st.header("📈 Scatter Plots & Regression")

    st.write("#### Drive-Wheels and Body-Style vs. Price")
    df_gptest = df[['drive-wheels', 'body-style', 'price']]
    grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
    grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)
    st.markdown("### Select Features for Regression Analysis")
    x_axis = st.selectbox("Choose the X-axis (numerical):", numerical_columns)
    y_axis = st.selectbox("Choose the Y-axis (numerical):", numerical_columns)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(grouped_pivot, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    # Scatter Plot
    st.write(f"#### Scatter Plot: {x_axis} vs. {y_axis}")
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_axis, y=y_axis, data=df, ax=ax, color="#3498db")
    st.pyplot(fig)

    st.write("#### Drive-Wheels vs. Price")
    grouped_drive = df[['drive-wheels', 'price']].groupby('drive-wheels', as_index=False).mean()
    # Regression Plot
    st.write(f"#### Regression Plot: {x_axis} vs. {y_axis}")
    fig, ax = plt.subplots()
    sns.heatmap(
        grouped_drive.set_index('drive-wheels').T,
        annot=True,
        fmt=".2f",
        cmap="RdBu",
        cbar=True,
        ax=ax,
    )
    sns.regplot(x=x_axis, y=y_axis, data=df, ax=ax, color="#e74c3c")
    st.pyplot(fig)

# Section 5: Key Findings
elif menu == "🔑 Key Findings":
    st.markdown('<h2 class="section-header">🔑 Key Findings</h2>', unsafe_allow_html=True)
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
        ### Key Insights:
        - **Engine Size**, **Horsepower**, and **Curb Weight** have a strong positive correlation with price.
        - **Highway MPG** and **City MPG** negatively impact price.
        - Categorical variables like **Drive Wheels** and **Body Style** significantly affect price distribution.
        ### What is a Correlation Matrix?
        - **Positive Correlation**: As one variable increases, so does the other.
        - **Negative Correlation**: As one variable increases, the other decreases.
        - **No Correlation**: No linear relationship exists.
        
        ### Conclusion:
        This dashboard provides a comprehensive analysis of key factors influencing car prices. Use these insights to drive business decisions or predictive modeling efforts.
        Use this matrix to identify relationships between variables, such as how engine size correlates with price.
        """
    )
    st.markdown('<div class="footer">Thank you for exploring! 🚗 Let’s drive data forward. 🌟</div>', unsafe_allow_html=True)

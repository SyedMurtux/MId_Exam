# Streamlit-based Python script for advanced automobile data analysis
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Page Configuration
st.set_page_config(
    page_title="🚗 Automobile Insights Dashboard",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Styling for Premium Look
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fa;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .main-header {
        font-size: 55px;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 40px;
    }
    .section-header {
        font-size: 36px;
        font-weight: bold;
        color: #34495e;
        margin-top: 30px;
        margin-bottom: 20px;
        text-align: center;
    }
    .highlight {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 8px;
        border-left: 6px solid #2980b9;
        color: #2c3e50;
        margin-bottom: 20px;
        font-size: 18px;
    }
    .footer {
        font-size: 14px;
        text-align: center;
        color: #7f8c8d;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-header">🚗 Automobile Insights Dashboard</div>', unsafe_allow_html=True)
st.markdown("### Dive into a world of interactive data analysis with stunning visuals and actionable insights! 🌟")

# Load Data
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(url)

df = load_data()

# Sidebar Navigation
st.sidebar.title("📂 Explore Sections")
menu = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 Overview",
        "📈 Regression Insights",
        "📊 Distribution & Box Plots",
        "🎨 Heatmaps",
        "🔑 Key Findings",
    ],
)

# Section 1: Overview
if menu == "🏠 Overview":
    st.markdown('<h2 class="section-header">🏠 Dataset Overview</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            '<div class="highlight">This dataset contains detailed attributes of cars, including their specifications and prices, allowing for in-depth analysis.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.write("### Dataset Info:")
        st.write(df.describe(include='all').T)

    st.markdown(
        '<div class="highlight">The dataset combines **categorical** and **numerical** data, perfect for exploring relationships between car specifications and price.</div>',
        unsafe_allow_html=True,
    )

# Section 2: Regression Insights
elif menu == "📈 Regression Insights":
    st.markdown('<h2 class="section-header">📈 Regression Insights</h2>', unsafe_allow_html=True)

    st.markdown(
        '<div class="highlight">Understand linear relationships between car attributes and price through scatter plots with regression lines.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Engine Size vs. Price")
        fig, ax = plt.subplots()
        sns.regplot(x="engine-size", y="price", data=df, ax=ax, color="#3498db")
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
        sns.regplot(x="peak-rpm", y="price", data=df, ax=ax, color="#2ecc71")
        st.pyplot(fig)

    with col4:
        st.write("#### Stroke vs. Price")
        fig, ax = plt.subplots()
        sns.regplot(x="stroke", y="price", data=df, ax=ax, color="#9b59b6")
        st.pyplot(fig)

# Section 3: Distribution & Box Plots
elif menu == "📊 Distribution & Box Plots":
    st.markdown('<h2 class="section-header">📊 Distribution & Box Plots</h2>', unsafe_allow_html=True)

    st.markdown(
        '<div class="highlight">Explore how car prices vary across categorical features like body style, engine location, and drive wheels.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Price Distribution by Body Style")
        fig, ax = plt.subplots()
        sns.boxplot(x="body-style", y="price", data=df, ax=ax, palette="coolwarm")
        st.pyplot(fig)

    with col2:
        st.write("#### Price Distribution by Engine Location")
        fig, ax = plt.subplots()
        sns.boxplot(x="engine-location", y="price", data=df, ax=ax, palette="Set3")
        st.pyplot(fig)

    st.write("#### Price Distribution by Drive Wheels")
    fig, ax = plt.subplots()
    sns.boxplot(x="drive-wheels", y="price", data=df, ax=ax, palette="Set2")
    st.pyplot(fig)

# Section 4: Heatmaps
elif menu == "🎨 Heatmaps":
    st.markdown('<h2 class="section-header">🎨 Heatmap Analysis</h2>', unsafe_allow_html=True)

    st.markdown(
        '<div class="highlight">Visualize the relationship between price and categorical attributes using interactive heatmaps.</div>',
        unsafe_allow_html=True,
    )

    st.write("#### Drive-Wheels and Body-Style vs. Price")
    df_gptest = df[['drive-wheels', 'body-style', 'price']]
    grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
    grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(grouped_pivot, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("#### Drive-Wheels vs. Price Heatmap")
    grouped_drive = df[['drive-wheels', 'price']].groupby('drive-wheels', as_index=False).mean()

    fig, ax = plt.subplots()
    sns.heatmap(
        grouped_drive.set_index('drive-wheels').T,
        annot=True,
        fmt=".2f",
        cmap="RdBu",
        cbar=True,
        ax=ax,
    )
    st.pyplot(fig)

# Section 5: Key Findings
elif menu == "🔑 Key Findings":
    st.markdown('<h2 class="section-header">🔑 Key Findings</h2>', unsafe_allow_html=True)

    st.markdown(
        """
        ### Key Insights:
        - **Engine Size**, **Horsepower**, and **Curb Weight** have a strong positive correlation with price.
        - **Highway MPG** and **City MPG** negatively impact price, suggesting efficiency-focused cars are less expensive.
        - **Drive Wheels** and **Body Style** play a significant role in price variation.
        
        ### Final Thoughts:
        This dashboard provides a comprehensive overview of car price determinants. Leverage these insights for predictive modeling or market strategy!
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="footer">Thank you for exploring this dashboard! 🚗</div>',
        unsafe_allow_html=True,
    )

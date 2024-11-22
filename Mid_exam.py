# Streamlit-based Python script for the ultimate automobile data analysis dashboard
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
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(url)

df = load_data()

# Sidebar Navigation
st.sidebar.title("🔍 Explore Sections")
menu = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 Overview",
        "📈 Regression Insights",
        "📊 Box Plot Analysis",
        "🎨 Interactive Heatmaps",
        "🔑 Key Findings",
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

    st.markdown(
        '<div class="highlight">Explore linear relationships between car attributes and price using regression plots.</div>',
        unsafe_allow_html=True,
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

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Price by Body Style")
        fig, ax = plt.subplots()
        sns.boxplot(x="body-style", y="price", data=df, ax=ax, palette="coolwarm")
        st.pyplot(fig)

    with col2:
        st.write("#### Price by Engine Location")
        fig, ax = plt.subplots()
        sns.boxplot(x="engine-location", y="price", data=df, ax=ax, palette="Set3")
        st.pyplot(fig)

    st.write("#### Price by Drive Wheels")
    fig, ax = plt.subplots()
    sns.boxplot(x="drive-wheels", y="price", data=df, ax=ax, palette="Set2")
    st.pyplot(fig)

# Section 4: Heatmaps
elif menu == "🎨 Interactive Heatmaps":
    st.markdown('<h2 class="section-header">🎨 Interactive Heatmaps</h2>', unsafe_allow_html=True)

    st.markdown(
        '<div class="highlight">Visualize price variations across categorical features using heatmaps.</div>',
        unsafe_allow_html=True,
    )

    st.write("#### Drive-Wheels and Body-Style vs. Price")
    df_gptest = df[['drive-wheels', 'body-style', 'price']]
    grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
    grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(grouped_pivot, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("#### Drive-Wheels vs. Price")
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
        - **Highway MPG** and **City MPG** negatively impact price.
        - Categorical variables like **Drive Wheels** and **Body Style** significantly affect price distribution.
        
        ### Conclusion:
        This dashboard provides a comprehensive analysis of key factors influencing car prices. Use these insights to drive business decisions or predictive modeling efforts.
        """
    )

    st.markdown('<div class="footer">Thank you for exploring! 🚗 Let’s drive data forward. 🌟</div>', unsafe_allow_html=True)

# Streamlit-based Python script for advanced automobile data analysis
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Page Configuration
st.set_page_config(
    page_title="🚗 The Ultimate Automobile Data Analysis",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f8fa;
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        font-size: 60px;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 30px;
        font-weight: bold;
        color: #34495e;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .highlight {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 6px solid #2980b9;
        font-size: 18px;
        color: #2c3e50;
    }
    .custom-footer {
        text-align: center;
        font-size: 14px;
        color: #7f8c8d;
        margin-top: 50px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-header">🚗 The Ultimate Automobile Data Analysis</div>', unsafe_allow_html=True)

st.markdown(
    "### Explore, Visualize, and Analyze car prices like never before. 🚘"
)

# Load Data
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(url)

df = load_data()

# Sidebar Navigation
st.sidebar.title("📂 Explore Sections")
menu = st.sidebar.radio(
    "Select a section:",
    [
        "🏠 Overview",
        "📈 Regression Insights",
        "📊 Box Plot Analysis",
        "🎨 Heatmap Visualizations",
        "🔑 Key Findings",
    ],
)

# Section 1: Overview
if menu == "🏠 Overview":
    st.markdown('<h2 class="sub-header">🏠 Dataset Overview</h2>', unsafe_allow_html=True)

    st.markdown(
        '<div class="highlight">This dataset contains details about cars, including their specifications and prices. Let’s dive in!</div>',
        unsafe_allow_html=True,
    )

    st.write("### Dataset Preview:")
    st.dataframe(df.head(), use_container_width=True)

    st.write("### Data Types:")
    st.write(df.dtypes)

    st.markdown(
        '<div class="highlight">The dataset includes both **categorical** and **numerical** variables, perfect for exploring relationships and drawing meaningful insights.</div>',
        unsafe_allow_html=True,
    )

# Section 2: Regression Insights
elif menu == "📈 Regression Insights":
    st.markdown('<h2 class="sub-header">📈 Regression Insights</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div class="highlight">Uncover how various features relate to car prices using scatter plots with regression lines.</div>',
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

# Section 3: Box Plot Analysis
elif menu == "📊 Box Plot Analysis":
    st.markdown('<h2 class="sub-header">📊 Box Plot Analysis</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div class="highlight">Discover how car prices vary across different categorical features using box plots.</div>',
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

# Section 4: Heatmap Visualizations
elif menu == "🎨 Heatmap Visualizations":
    st.markdown('<h2 class="sub-header">🎨 Heatmap Visualizations</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div class="highlight">Explore how price is influenced by multiple factors using heatmaps.</div>',
        unsafe_allow_html=True,
    )

    st.write("#### Heatmap: Drive-Wheels and Body-Style vs. Price")
    df_gptest = df[['drive-wheels', 'body-style', 'price']]
    grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
    grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(grouped_pivot, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("#### Heatmap: Drive-Wheels vs. Price")
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
    st.markdown('<h2 class="sub-header">🔑 Key Findings</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        ### Key Takeaways:
        - **Engine Size**, **Horsepower**, and **Curb Weight** strongly correlate with car prices.
        - Features like **Highway MPG** and **City MPG** have a negative relationship with price.
        - Categorical variables like **Drive Wheels** and **Body Style** significantly influence price.
        
        ### Insights for Action:
        Use these findings to build predictive models or make strategic decisions in the automobile market!
        """
    )

    st.markdown(
        '<div class="custom-footer">🚗 Thank you for exploring! Dive deeper into your automobile analysis journey. 🚀</div>',
        unsafe_allow_html=True,
    )

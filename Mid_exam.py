# Streamlit-based Python script for analyzing automobile data
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Page Configuration
st.set_page_config(
    page_title="🚗 Automobile Data Analysis",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f7f7;
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        font-size: 50px;
        font-weight: bold;
        color: #444444;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 30px;
        font-weight: bold;
        color: #333333;
        margin-top: 20px;
    }
    .highlight {
        background-color: #fff4e6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ffa500;
        font-weight: bold;
        color: #444444;
    }
    .custom-footer {
        text-align: center;
        font-size: 14px;
        color: #888888;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.markdown('<div class="main-header">🚗 Automobile Data Analysis</div>', unsafe_allow_html=True)

st.markdown(
    "Welcome to the interactive automobile data analysis tool! 🎨 Explore fascinating insights into car prices, analyze patterns, and understand key factors that influence the automobile market."
)

# Load Data
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(url)

df = load_data()

# Sidebar Navigation
st.sidebar.title("📂 Navigation")
menu = st.sidebar.radio(
    "Choose a Section",
    [
        "🏠 Overview",
        "📈 Regression Plots",
        "📊 Box Plots",
        "🎨 Heat Maps",
        "📋 Conclusions",
    ],
)

# Section 1: Overview
if menu == "🏠 Overview":
    st.markdown('<h2 class="sub-header">🏠 Dataset Overview</h2>', unsafe_allow_html=True)

    st.write("### Dataset Preview:")
    st.dataframe(df.head(), use_container_width=True)

    st.write("### Data Types:")
    st.write(df.dtypes)

    st.markdown(
        '<div class="highlight">This dataset contains numerical and categorical variables to explore, analyze, and visualize key factors affecting car prices.</div>',
        unsafe_allow_html=True,
    )

# Section 2: Regression Plots
elif menu == "📈 Regression Plots":
    st.markdown('<h2 class="sub-header">📈 Regression Analysis</h2>', unsafe_allow_html=True)
    st.write("Explore relationships between various variables and car prices with linear regression.")

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Engine Size vs. Price")
        fig, ax = plt.subplots()
        sns.regplot(x="engine-size", y="price", data=df, ax=ax, color="#1f77b4")
        st.pyplot(fig)

    with col2:
        st.write("#### Highway MPG vs. Price")
        fig, ax = plt.subplots()
        sns.regplot(x="highway-mpg", y="price", data=df, ax=ax, color="#ff7f0e")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.write("#### Peak RPM vs. Price")
        fig, ax = plt.subplots()
        sns.regplot(x="peak-rpm", y="price", data=df, ax=ax, color="#2ca02c")
        st.pyplot(fig)

    with col4:
        st.write("#### Stroke vs. Price")
        fig, ax = plt.subplots()
        sns.regplot(x="stroke", y="price", data=df, ax=ax, color="#d62728")
        st.pyplot(fig)

# Section 3: Box Plots
elif menu == "📊 Box Plots":
    st.markdown('<h2 class="sub-header">📊 Box Plot Analysis</h2>', unsafe_allow_html=True)
    st.write("Analyze the distribution of car prices across categorical variables.")

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

# Section 4: Heat Maps
elif menu == "🎨 Heat Maps":
    st.markdown('<h2 class="sub-header">🎨 Heat Map Analysis</h2>', unsafe_allow_html=True)

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

# Section 5: Conclusion
elif menu == "📋 Conclusions":
    st.markdown('<h2 class="sub-header">📋 Conclusions</h2>', unsafe_allow_html=True)

    st.markdown(
        """
        ### Key Takeaways:
        - Continuous variables such as `engine-size`, `horsepower`, and `curb-weight` have a strong positive correlation with price.
        - Negative correlations exist for `highway-mpg` and `city-mpg` with price.
        - Categorical variables like `drive-wheels` and `body-style` significantly influence price categories.
        
        ### Final Thoughts:
        This interactive tool highlights the key factors that impact car prices, providing valuable insights for data analysis and predictive modeling.
        """
    )

    st.markdown(
        '<div class="custom-footer">Thank you for exploring! 🚗✨</div>',
        unsafe_allow_html=True,
    )

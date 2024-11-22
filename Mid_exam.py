# Streamlit-based Python script for analyzing automobile data
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Page Config
st.set_page_config(
    page_title="Automobile Data Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Data
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(url)

df = load_data()

# Sidebar Navigation
st.sidebar.header("Navigation")
menu = st.sidebar.radio(
    "Choose a Section",
    [
        "📊 Dataset Overview",
        "📈 Regression Plots",
        "📉 Box Plots",
        "🎨 Heat Maps",
        "📋 Conclusion",
    ],
)

# Section 1: Dataset Overview
if menu == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    st.write("### Data Types")
    st.write(df.dtypes)

# Section 2: Regression Plots
elif menu == "📈 Regression Plots":
    st.title("📈 Regression Plots")
    st.write("#### 1. Engine Size vs. Price")
    fig, ax = plt.subplots()
    sns.regplot(x="engine-size", y="price", data=df, ax=ax, color="blue")
    st.pyplot(fig)

    st.write("#### 2. Highway MPG vs. Price")
    fig, ax = plt.subplots()
    sns.regplot(x="highway-mpg", y="price", data=df, ax=ax, color="green")
    st.pyplot(fig)

    st.write("#### 3. Peak RPM vs. Price")
    fig, ax = plt.subplots()
    sns.regplot(x="peak-rpm", y="price", data=df, ax=ax, color="orange")
    st.pyplot(fig)

    st.write("#### 4. Stroke vs. Price")
    fig, ax = plt.subplots()
    sns.regplot(x="stroke", y="price", data=df, ax=ax, color="red")
    st.pyplot(fig)

# Section 3: Box Plots
elif menu == "📉 Box Plots":
    st.title("📉 Box Plots")
    st.write("#### 1. Price Distribution by Body Style")
    fig, ax = plt.subplots()
    sns.boxplot(x="body-style", y="price", data=df, ax=ax, palette="Set2")
    st.pyplot(fig)

    st.write("#### 2. Price Distribution by Engine Location")
    fig, ax = plt.subplots()
    sns.boxplot(x="engine-location", y="price", data=df, ax=ax, palette="Set3")
    st.pyplot(fig)

    st.write("#### 3. Price Distrib

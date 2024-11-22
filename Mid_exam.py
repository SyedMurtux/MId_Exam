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

    st.write("#### 3. Price Distribution by Drive Wheels")
    fig, ax = plt.subplots()
    sns.boxplot(x="drive-wheels", y="price", data=df, ax=ax, palette="coolwarm")
    st.pyplot(fig)

# Section 4: Heat Maps
elif menu == "🎨 Heat Maps":
    st.title("🎨 Heat Maps")
    st.write("#### 1. Drive-Wheels and Body-Style vs. Price Heatmap")
    df_gptest = df[['drive-wheels', 'body-style', 'price']]
    grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
    grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(grouped_pivot, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("#### 2. Drive-Wheels vs. Price Heatmap")
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
elif menu == "📋 Conclusion":
    st.title("📋 Conclusion")
    st.markdown(
        """
        **Key Takeaways:**
        - Engine size, horsepower, and curb weight show strong positive correlations with price.
        - Highway-mpg and city-mpg have negative correlations with price.
        - Drive-wheels and body-style play important roles in price categorization.
        
        This analysis highlights the key variables that influence car prices and provides insights for predictive modeling.
        """
    )

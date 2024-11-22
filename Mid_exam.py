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

# Custom CSS for better visuals
st.markdown(
    """
    <style>
    .stApp { 
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #333333;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #444444;
        margin-top: 20px;
    }
    .highlight {
        background-color: #fffae6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.markdown('<h1 class="main-header">🚗 Automobile Data Analysis</h1>', unsafe_allow_html=True)
st.markdown(
    "Dive into the fascinating world of automobile data and explore what influences car prices. This interactive app provides insights, visualizations, and statistical analysis in a creative and engaging way."
)

# Load Data
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(url)

df = load_data()

# Sidebar
st.sidebar.header("Navigation")
menu = st.sidebar.radio(
    "Choose a Section",
    [
        "📊 Dataset Overview",
        "📈 Correlation Analysis",
        "📉 Regression Analysis",
        "📂 Grouped Insights",
        "🎨 Visualizations",
        "📋 Conclusion",
    ],
)

# Section 1: Dataset Overview
if menu == "📊 Dataset Overview":
    st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    st.write("### Data Types:")
    st.write(df.dtypes)

    st.write(
        '<div class="highlight">The dataset contains various attributes related to cars, including numerical and categorical variables, which we will analyze in depth.</div>',
        unsafe_allow_html=True,
    )

# Section 2: Correlation Analysis
elif menu == "📈 Correlation Analysis":
    st.markdown('<h2 class="sub-header">📈 Correlation Analysis</h2>', unsafe_allow_html=True)
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()

    st.write("### Correlation Matrix for Numerical Variables:")
    st.dataframe(correlation_matrix)

    st.write(
        '<div class="highlight">Higher positive or negative values in the matrix indicate a strong relationship between variables.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Selected Features Correlation Matrix")
    selected_columns = ['bore', 'stroke', 'compression-ratio', 'horsepower']
    selected_corr = df[selected_columns].corr()
    st.dataframe(selected_corr)

# Section 3: Regression Analysis
elif menu == "📉 Regression Analysis":
    st.markdown('<h2 class="sub-header">📉 Regression Analysis</h2>', unsafe_allow_html=True)
    st.write("### Engine Size vs. Price")
    fig, ax = plt.subplots()
    sns.regplot(x="engine-size", y="price", data=df, ax=ax, color="#1f77b4")
    st.pyplot(fig)

    st.write("### Stroke vs. Price")
    fig, ax = plt.subplots()
    sns.regplot(x="stroke", y="price", data=df, ax=ax, color="#ff7f0e")
    st.pyplot(fig)

# Section 4: Grouped Insights
elif menu == "📂 Grouped Insights":
    st.markdown('<h2 class="sub-header">📂 Grouped Insights</h2>', unsafe_allow_html=True)

    # Average Price by Body Style
    st.write("### Average Price by Body Style")
    df_grouped = df[['body-style', 'price']]
    df_grouped['price'] = pd.to_numeric(df_grouped['price'], errors='coerce')
    avg_price_body = df_grouped.groupby('body-style', as_index=False)['price'].mean()
    st.dataframe(avg_price_body)

    # Heatmap
    st.write("### Heatmap: Drive-Wheels and Body-Style vs. Price")
    df_gptest = df[['drive-wheels', 'body-style', 'price']]
    grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
    grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(grouped_pivot, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Section 5: Visualizations
elif menu == "🎨 Visualizations":
    st.markdown('<h2 class="sub-header">🎨 Visualizations</h2>', unsafe_allow_html=True)

    st.write("### Price Distribution by Body Style")
    fig, ax = plt.subplots()
    sns.boxplot(x="body-style", y="price", data=df, ax=ax, palette="Set2")
    st.pyplot(fig)

    st.write("### Price Distribution by Engine Location")
    fig, ax = plt.subplots()
    sns.boxplot(x="engine-location", y="price", data=df, ax=ax, palette="Set3")
    st.pyplot(fig)

# Section 6: Conclusion
elif menu == "📋 Conclusion":
    st.markdown('<h2 class="sub-header">📋 Conclusion</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        **Key Findings:**
        - Continuous variables such as `engine-size`, `curb-weight`, and `width` have a strong correlation with car price.
        - Categorical variables like `drive-wheels` and `body-style` also influence car price significantly.
        
        This analysis provides a strong foundation for predictive modeling and understanding price-driving factors in the automobile industry.
        """
    )
    st.markdown('<div class="highlight">We hope you enjoyed this analysis. 🚗✨</div>', unsafe_allow_html=True)

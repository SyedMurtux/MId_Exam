# Streamlit-based Python script for analyzing automobile data
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Page Title
st.title("Automobile Data Analysis and Visualization")
st.markdown("""
Analyze the characteristics that impact car prices using interactive data visualizations and statistical measures.
""")

# Load Data
st.sidebar.header("Dataset")
st.sidebar.write("The dataset used in this analysis is hosted on GitHub.")
data_url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'

# Load the data
st.sidebar.subheader("Data Preview")
df = pd.read_csv(data_url)
st.write("### Dataset Preview", df.head())

# Data Types Section
st.write("### Data Types")
st.write("The following are the data types of the columns in the dataset:")
st.write(df.dtypes)

# Correlation Analysis
st.write("### Correlation Analysis")
st.write("""
We can calculate the correlation between numerical variables using the Pearson Correlation method.
""")
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
st.write("Correlation Matrix:")
st.dataframe(correlation_matrix)

# Visualization: Engine Size vs. Price
st.write("### Scatter Plot: Engine Size vs. Price")
st.write("""
The scatter plot below shows the relationship between Engine Size and Price.
""")
fig, ax = plt.subplots()
sns.regplot(x="engine-size", y="price", data=df, ax=ax)
st.pyplot(fig)

# Correlation Matrix for Selected Columns
st.write("### Selected Features Correlation Matrix")
st.write("Find the correlation between 'bore', 'stroke', 'compression-ratio', and 'horsepower'.")
selected_columns = ['bore', 'stroke', 'compression-ratio', 'horsepower']
selected_df = df[selected_columns]
correlation_matrix_selected = selected_df.corr()
st.dataframe(correlation_matrix_selected)

# Scatter Plot: Stroke vs. Price
st.write("### Scatter Plot: Stroke vs. Price")
fig, ax = plt.subplots()
sns.regplot(x="stroke", y="price", data=df, ax=ax)
st.pyplot(fig)

# Grouped Analysis
st.write("### Grouped Analysis")
st.write("""
Group data by `body-style` and calculate the average price.
""")
df_grouped = df[['body-style', 'price']]
df_grouped['price'] = pd.to_numeric(df_grouped['price'], errors='coerce')
average_price_by_body_style = df_grouped.groupby('body-style', as_index=False)['price'].mean()
st.write("Average Price by Body Style:")
st.dataframe(average_price_by_body_style)

# Heatmap Visualization
st.write("### Heatmap: Drive-Wheels and Body-Style vs. Price")
df_gptest = df[['drive-wheels', 'body-style', 'price']]
grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(grouped_pivot, annot=True, fmt=".2f", cmap='RdBu', ax=ax)
st.pyplot(fig)

# Pearson Correlation Analysis with P-value
st.write("### Pearson Correlation with P-values")
st.write("""
The following table shows the Pearson Correlation Coefficient and P-values for selected variables with respect to price.
""")

correlation_results = []
variables = ['wheel-base', 'horsepower', 'length', 'width', 'curb-weight', 'engine-size', 'bore', 'city-mpg', 'highway-mpg']

for var in variables:
    coef, p_value = stats.pearsonr(df[var], df['price'])
    correlation_results.append({'Variable': var, 'Correlation Coefficient': coef, 'P-value': p_value})

correlation_results_df = pd.DataFrame(correlation_results)
st.dataframe(correlation_results_df)

# Conclusion Section
st.write("### Conclusion")
st.write("""
After analyzing the dataset, we identified the following characteristics as having the most significant impact on car prices:

#### Continuous Numerical Variables:
- Length
- Width
- Curb-weight
- Engine-size
- Horsepower
- City-mpg
- Highway-mpg
- Wheel-base
- Bore

#### Categorical Variables:
- Drive-wheels

These variables can be further used in predictive modeling to estimate car prices.
""")

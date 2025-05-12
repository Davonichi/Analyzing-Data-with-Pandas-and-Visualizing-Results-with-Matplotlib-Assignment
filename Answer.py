# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load dataset from sklearn
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    # Clean dataset (no missing values, but here's how you would drop them)
    df_cleaned = df.dropna()
except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis

# Basic statistics
print("\nDescriptive statistics:")
print(df_cleaned.describe())

# Grouping by species and calculating the mean
print("\nMean values grouped by species:")
grouped_means = df_cleaned.groupby('species').mean()
print(grouped_means)

# Pattern finding
print("\nInteresting Finding:")
print("→ Iris-virginica tends to have the longest petals and sepals on average.")

# Task 3: Data Visualization

# Set Seaborn style
sns.set(style="whitegrid")

# Line Chart: Simulating a time series (fake index for demo)
plt.figure(figsize=(10, 5))
plt.plot(df_cleaned.index, df_cleaned['sepal length (cm)'], label='Sepal Length')
plt.title("Line Chart: Sepal Length Over Sample Index")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# Bar Chart: Average petal length per species
plt.figure(figsize=(8, 6))
sns.barplot(x=grouped_means.index, y=grouped_means['petal length (cm)'], palette="viridis")
plt.title("Bar Chart: Avg Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# Histogram: Distribution of petal length
plt.figure(figsize=(8, 6))
sns.histplot(df_cleaned['petal length (cm)'], bins=20, kde=True, color="orange")
plt.title("Histogram: Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot: Sepal length vs Petal length
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_cleaned, x='sepal length (cm)', y='petal length (cm)', hue='species', palette="deep")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.show()

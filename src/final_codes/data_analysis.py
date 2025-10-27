import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from word2number import w2n
from tqdm import tqdm

# =======================
# Load and Prepare Data
# =======================
df = pd.read_excel(r"\database\Cleaning_Final_Clean_Project_Data.xlsx")

# Fill missing values with 0
df.fillna(0, inplace=True)

# Select numeric columns
numeric_df = df.select_dtypes(include=["float64", "int64"])

# =======================
# Overall Correlation Heatmap
# =======================
plt.figure(figsize=(20, 20))
sns.heatmap(numeric_df.corr(), annot=True, cmap="RdYlGn")
plt.title("Correlation Heatmap for Numeric Features")
plt.show()

# =======================
# Correlation of Price with All Numeric Features
# =======================
if "Price" in numeric_df.columns:
    price_corr = numeric_df.corr()["Price"].sort_values(ascending=False)

    # 1️⃣ Horizontal Heatmap of All Correlations
    plt.figure(figsize=(20, 1))
    sns.heatmap(price_corr.to_frame().T, annot=True, fmt=".2f", cmap="RdYlGn")
    plt.title("Correlation of Price with Other Features")
    plt.show()

    # 2️⃣ Bar Plot (All)
    plt.figure(figsize=(10, 6))
    price_corr.plot.bar(color="skyblue")
    plt.axhline(y=0.5, color="r", linestyle="--", linewidth=1)
    plt.title("Correlation with Price (All Features)")
    plt.ylabel("Correlation Coefficient")
    plt.show()

    # 3️⃣ Filtered Bar Plot (|corr| > 0.08)
    filtered_corr = price_corr[price_corr.abs() > 0.08]
    plt.figure(figsize=(10, 6))
    filtered_corr.plot.bar(color="royalblue")
    plt.axhline(y=0.5, color="r", linestyle="--", linewidth=1)
    plt.title("Filtered Correlation with Price (|corr| > 0.08)")
    plt.ylabel("Correlation Coefficient")
    plt.xlabel("Features")
    plt.xticks(rotation=45)
    plt.show()

    # 4️⃣ Filtered Heatmap
    plt.figure(figsize=(20, 1))
    sns.heatmap(filtered_corr.to_frame().T, annot=True, fmt=".2f", cmap="RdYlGn")
    plt.title("Correlation of Price with Key Features (|corr| > 0.08)")
    plt.show()
else:
    print("Column 'Price' not found in dataset.")

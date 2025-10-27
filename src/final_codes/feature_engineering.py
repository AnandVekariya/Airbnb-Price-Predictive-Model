import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from word2number import w2n
from tqdm import tqdm

# ==========================
# Load and Inspect Data
# ==========================
file_path = r"\database\Cleaning_Final_Clean_Project_Data.xlsx"

df = pd.read_excel(file_path)
print("✅ Data loaded successfully!")
print("Data shape:", df.shape)
print(df.head())

# ==========================
# Identify Non-Numeric Columns
# ==========================
non_numeric_cols = df.select_dtypes(exclude=["int64", "float64"]).columns
print("\nNon-numeric columns:")
print(non_numeric_cols.tolist())

# ==========================
# One-Hot Encode Categorical Columns
# ==========================
# List of columns to convert into dummy/indicator variables
columns_to_encode = [
    "Host Response Time",
    "Country",
    "Property Type",
    "Room Type",
    "Bed Type",
    "Cancellation Policy",
]

# Use pandas built-in get_dummies (faster and cleaner than manual loops)
df_encoded = pd.get_dummies(df, columns=columns_to_encode, prefix=columns_to_encode)

print("\n✅ One-hot encoding completed!")
print("New shape after encoding:", df_encoded.shape)

# ==========================
# Save the Processed Data
# ==========================
output_path = r"\database\Modeling_Final_Clean_Project_Data.xlsx"
df_encoded.to_excel(output_path, index=False)

print(f"\n💾 Cleaned and encoded data saved successfully to:\n{output_path}")

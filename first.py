
import pandas as pd

# Load the dataset
df = pd.read_csv("data/ai4i2020.csv")

# Display first 5 rows
print("FIRST 5 ROWS:")
print(df.head())

# Display number of rows and columns
print("\nDATASET SHAPE:")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

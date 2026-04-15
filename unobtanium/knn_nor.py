import numpy as np
import pandas as pd

# Load your data
source1 = pd.read_csv('data/mag-kiruna-compiled/smooth.csv', delimiter=";")
source2 = pd.read_csv('data/solarwinds-ace-compiled/smooth.csv', delimiter=";")
source3 = pd.read_csv('data/kp-compiled/smooth.csv', delimiter=";")

# Function to normalize all numeric columns except those in exclude_columns
def normalize_to_01_except(df, exclude_columns):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude_columns)
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    return df

# Exclude columns you don't want to normalize
exclude_source1 = ['Date']  # Add more columns if needed
exclude_source2 = ['Date']  # Add more columns if needed

# Normalize and save
# source1_normalized = normalize_to_01_except(source1, exclude_source1)
# source2_normalized = normalize_to_01_except(source2, exclude_source2)

data = pd.merge(source1, source2, "inner", "Date")
data = pd.merge(data, source3, "inner", "Date")



# Save to new files
data.to_csv('data/mag-kiruna-compiled/smooth_normalized2.csv', sep=";", index=False)
# source2_normalized.to_csv('data/solarwinds-ace-compiled/smooth_normalized.csv', sep=";", index=False)

print("Normalized data saved to:")
print("- data/mag-kiruna-compiled/smooth_normalized.csv")
print("- data/solarwinds-ace-compiled/smooth_normalized.csv")
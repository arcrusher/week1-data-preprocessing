"""
Week 1 Internship Task: Data Acquisition and Pre-processing in Python
Dataset: Iris Flower Dataset (scikit-learn built-in public dataset)

Run:
    pip install pandas numpy scikit-learn matplotlib seaborn
    python week1_iris_preprocessing.py

Outputs:
    iris_raw.csv
    iris_cleaned.csv
    iris_pairplot.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

# 1. DATA ACQUISITION
# The Iris dataset is a publicly available classic dataset distributed
# with scikit-learn. This loader acquires the data without a separate download.
iris = load_iris(as_frame=True)
df = iris.data.copy()
df["species"] = iris.target_names[iris.target]
df.to_csv("iris_raw.csv", index=False)

print("Dataset acquired successfully.")
print("Shape:", df.shape)
print(df.head())

# 2. INITIAL INSPECTION
print("\\nData types:\\n", df.dtypes)
print("\\nMissing values:\\n", df.isnull().sum())
print("\\nDuplicate rows:", df.duplicated().sum())
print("\\nSummary statistics:\\n", df.describe())

# 3. DATA CLEANING
# Remove exact duplicate rows if any.
df = df.drop_duplicates().reset_index(drop=True).copy()

# Separate numerical features and target.
feature_cols = iris.feature_names
X = df[feature_cols]
y = df["species"]

# Impute missing numerical values with the median. The original Iris data
# has no missing values; this step makes the pipeline robust to future data.
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

# Detect potential outliers with the IQR rule and cap them at the IQR fences.
# This avoids deleting rows and keeps the sample size stable.
X_clean = X_imputed.copy()
outlier_summary = {}
for col in feature_cols:
    q1 = X_clean[col].quantile(0.25)
    q3 = X_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((X_clean[col] < lower) | (X_clean[col] > upper)).sum())
    outlier_summary[col] = count
    X_clean[col] = X_clean[col].clip(lower, upper)

print("\\nPotential outliers flagged by IQR:", outlier_summary)

# 4. FEATURE SCALING
standard_scaler = StandardScaler()
X_standard = pd.DataFrame(
    standard_scaler.fit_transform(X_clean),
    columns=feature_cols
)

minmax_scaler = MinMaxScaler()
X_minmax = pd.DataFrame(
    minmax_scaler.fit_transform(X_clean),
    columns=feature_cols
)

# 5. FEATURE SELECTION
selector = SelectKBest(score_func=f_classif, k=3)
X_selected_array = selector.fit_transform(X_standard, y)
selected_features = list(np.array(feature_cols)[selector.get_support()])
print("\\nSelected top 3 features:", selected_features)

# 6. FINAL DATASET
final_df = X_clean.copy()
final_df["species"] = y.values
final_df.to_csv("iris_cleaned.csv", index=False)
print("\\nCleaned dataset saved as iris_cleaned.csv")
print("Final shape:", final_df.shape)

# 7. VISUALIZATION
# Simple scatter plot of sepal length vs petal length by species.
plt.figure(figsize=(8, 5))
for species in sorted(y.unique()):
    mask = y == species
    plt.scatter(
        X_clean.loc[mask, "sepal length (cm)"],
        X_clean.loc[mask, "petal length (cm)"],
        label=species,
        alpha=0.75
    )
plt.xlabel("Sepal length (cm)")
plt.ylabel("Petal length (cm)")
plt.title("Iris Dataset: Sepal Length vs Petal Length")
plt.legend(title="Species")
plt.tight_layout()
plt.savefig("iris_pairplot.png", dpi=150)
plt.close()

print("Visualization saved as iris_pairplot.png")
print("\\nPipeline completed successfully.")

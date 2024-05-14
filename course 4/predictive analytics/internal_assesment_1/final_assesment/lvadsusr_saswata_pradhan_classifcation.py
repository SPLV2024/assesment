# -*- coding: utf-8 -*-
"""LVADSUSR_SASwata_pradhan_Classifcation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Gkvm4OXLekNw8heDVzSR8_Il70C7zI9R
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split
import pandas as pd

url = 'https://raw.githubusercontent.com/Deepsphere-AI/LVA-Batch5-Assessment/main/penguins_classification.csv'
df= pd.read_csv(url)

# Print the DataFrame
df

df.drop('year',axis=1,inplace=True)

# Drop rows with NaN values
df_dropna = df.dropna()

# Plot histograms for each numeric column
plt.figure(figsize=(12, 8))
for i, column in enumerate(df_dropna.columns):
    plt.subplot(3, 4, i + 1) #
    sns.histplot(df_dropna[column], kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()

from sklearn.impute import SimpleImputer
import numpy as np

# Define a custom imputer function
def custom_imputer(column):
    if column.skew() > 1 or column.skew() < -1:
        return column.median()
    else:
        return column.mean()

# Apply custom imputer to each column
def impute_with_custom(df):
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(custom_imputer(df[col]), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# Impute using the custom imputer function
df_imputed = impute_with_custom(df)

# Check if there are any missing values left
print(df_imputed.isnull().sum())

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'species', y = 'bill_length_mm', data = df)

target=df['species']

# Assuming 'quality' is the target column, categorize it into "bad", "neutral", and "good" labels
if target.dtype == 'object':
    # If the target column is categorical, no binning needed
    labels = target

# Check the distribution of the quality labels
print("Class Distribution Before SMOTE:")
print(target.value_counts())

# Encode quality labels
label_quality = LabelEncoder()
target = label_quality.fit_transform(target)

# Separate the dataset into feature variables (X) and the response variable (y)
X = df.drop('species', axis=1)
y = target

# Encode object columns only in feature variables
object_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}
for col in object_cols:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

# Perform standard scaling only on numeric columns
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])



# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train RandomForestClassifier on the resampled data
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Predictions
pred_rfc = rfc.predict(X_test)

# Model performance
print("\nClassification Report:")
print(classification_report(y_test, pred_rfc))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_rfc))

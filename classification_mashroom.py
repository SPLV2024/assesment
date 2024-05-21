
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
dataset = pd.read_csv('/content/mushroom.csv')

# Exploratory Data Analysis (EDA)
# Display the first five rows of the dataset
print("First five rows of the dataset:")
print(dataset.head())

# Check for null values in the dataset
print("Checking for null values:")
print(dataset.isnull().sum().sum())

# Display the unique values of the target variable
print("Unique values in the target variable 'class':")
print(dataset['class'].unique())

# Display dataset information
print("Dataset information:")
print(dataset.info())

# Display the shape of the dataset
print("Shape of the dataset (rows, columns):")
print(dataset.shape)

# Plot the distribution of the target variable 'class'
sns.histplot(dataset['class'])
plt.title('Distribution of Edible vs Poisonous Mushrooms')
plt.show()

# Fill missing values with the median of each column
dataset = dataset.fillna(dataset.median())

# Separate features and target variable
X = dataset.drop(['class'], axis=1)
y = dataset['class']

# Convert categorical features to dummy variables
X = pd.get_dummies(X)
print("First five rows of the transformed features:")
print(X.head())

# Encode the target variable
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print("Encoded target variable:")
print(y)

"""#Feature Selection"""

# Split the dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("Shape of training set (X_train, y_train):", X_train.shape, y_train.shape)
print("Shape of test set (X_test, y_test):", X_test.shape, y_test.shape)

# Decision Tree Creation using Gini Index
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

# Fit the model
clf_gini.fit(X_train, y_train)

import matplotlib.pyplot as plt
from sklearn import tree

# Convert class names to strings
class_names = [str(cls) for cls in encoder.classes_]

# Plot the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf_gini, filled=True, feature_names=X.columns, class_names=class_names)
plt.title('Decision Tree using Gini Index')
plt.show()

# Predict the values for the test set
y_pred_gini = clf_gini.predict(X_test)

# Predict the values for the training set (for accuracy comparison)
y_pred_train_gini = clf_gini.predict(X_train)

# Determine the accuracy score
test_accuracy = accuracy_score(y_test, y_pred_gini)
train_accuracy = accuracy_score(y_train, y_pred_train_gini)

print('Model accuracy score with criterion gini index: {:.4f}'.format(test_accuracy))
print('Training-set accuracy score: {:.4f}'.format(train_accuracy))

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Convert class names to strings
class_names = [str(cls) for cls in encoder.classes_]

# Additional Evaluation Metrics
print("Classification Report for Test Set:")
print(classification_report(y_test, y_pred_gini, target_names=class_names))

print("Confusion Matrix for Test Set:")
conf_matrix = confusion_matrix(y_test, y_pred_gini)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Test Set')
plt.show()


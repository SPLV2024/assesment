

import pandas as pd

data = pd.read_csv('/content/bengaluru_house_prices.csv')
print(data.shape)
data.head()

"""#Data Pre-processing

Handle Missing Values
"""

# Checking for missing values
data.isnull().sum()

# Visualize missing values
import matplotlib.pyplot as plt

plt.figure(figsize=(15,7))
plt.barh(data.isnull().sum().index, width=data.isnull().sum().values)
plt.xlabel('Total Null Values')
plt.show()

# Fill missing values in 'balcony' and other columns
data['balcony'].replace({0.0:1,1.0:2,2.0:3,3.0:4}, inplace=True)
data.fillna(0, inplace=True)
data.isnull().sum()

"""#Handle Outliers"""

# Visualizing outliers
import seaborn as sns

plt.figure(figsize=(15,7))
sns.boxplot(data=data, x='area_type', y='price')
plt.show()

sub_data = data[data['area_type'] == 'Plot  Area']
plt.figure(figsize=(15,7))
sns.boxplot(data=sub_data, x='balcony', y='price')
plt.show()

plt.figure(figsize=(15,7))
sns.boxplot(data=sub_data, x='bath', y='price')
plt.show()

"""Encode Categorical Data"""

cat_columns = ['area_type', 'availability', 'location', 'size', 'society', 'bath', 'balcony']
for col in cat_columns:
    data[col] = data[col].astype('category')
    data[col] = data[col].cat.codes

"""Convert 'total_sqft' to Numeric"""

import re
import numpy as np

data['total_sqft'] = data['total_sqft'].apply(lambda x: x.split(' - ')[1] if len(x.split(' - ')) > 1 else x.split(' - ')[0])
data['total_sqft'] = data['total_sqft'].apply(lambda x: re.findall(r'\d+', x)[0])
data['total_sqft'] = data['total_sqft'].astype(np.float64)

"""#Exploratory Data Analysis
Descriptive Statistics
"""

data.describe()

data.shape

# Distribution of Price
plt.figure(figsize=(15,7))
sns.distplot(data['price'])
plt.show()

# Boxplot of Price by Area Type
plt.figure(figsize=(15,7))
sns.boxplot(data=data, x='area_type', y='price')
plt.show()

"""#Model Training & Testing"""

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=41)
y_train = train['price']
x_train = train.drop('price', axis=1)
y_test = test['price']
x_test = test.drop('price', axis=1)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

"""#Model Evaluation Metrics"""

from sklearn.metrics import mean_squared_error
import numpy as np

prediction = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, prediction))
print("RMSE:", rmse)

# Coefficients and Intercept
model.coef_, model.intercept_

# Visualize Predictions vs Actual
plt.figure(figsize=(15,7))
sns.distplot(y_test, color='r', label='Actual')
sns.distplot(prediction, label='Predicted')
plt.legend()
plt.show()

# Plot between first hundred "total area in sqft" and actual output
plt.figure(figsize=(15,7))
plt.plot(x_test['total_sqft'][:100], x_test['total_sqft'][:100] * model.coef_[5] + model.intercept_, label='Predicted Line')
plt.scatter(x_test['total_sqft'][:100], y_test[:100], label='Actual Data')
plt.legend()
plt.show()



"""Business Recommendations

Price Prediction Accuracy: With an RMSE of 108.26, the model provides a decent estimate of house prices. However, improving the model's accuracy by including more features or using advanced techniques like ensemble methods could yield better results.

Market Strategy: Real estate developers and agents can use the model to set competitive prices for properties by factoring in key attributes such as area type, total square footage, and the number of bathrooms and balconies.

Investment Decisions: Investors can leverage the price predictions to identify undervalued properties and make informed investment choices.

Feature Importance: Understanding which features most significantly impact house prices can guide renovation and development decisions to maximize property value.

Further Analysis: Continuously updating the model with new data will help maintain its accuracy and relevance in a dynamic market.
"""


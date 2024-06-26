# -*- coding: utf-8 -*-
"""LVADUSR-Saswata_Pradhan.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fUhChgm7Idb7vDEj1GzkNAjw4knKk4D9

1
"""

import numpy as np

data = np.array([[30, 170, 70], [40, 180, 80], [25, 160, 60]])

def data_ops(data):
    min=np.min(data)
    max=np.max(data)
    sum_=np.sum(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return min,max,sum_,mean,std
data_ops(data)

"""2.
```
health_data = np.array([[160, 70, 30],   # height, weight, age for individual 1
                        [165, 65, 35],   # height, weight, age for individual 2
                        [170, 75, 40]])  # height, weight, age for individual 3
```
"""

health_data = np.array([[160, 70, 30],   # height, weight, age for individual 1
                        [165, 65, 35],   # height, weight, age for individual 2
                        [170, 75, 40]])  # height, weight, age for individual 3
health_data


def normalizee(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data

health_data = np.array([[30, 170, 70], [40, 180, 80], [25, 160, 60]])
normalized_data = normalizee(health_data)
print(normalized_data)

"""3."""

import pandas as pd

# Load the dataset of property listings from CSV file
url = 'https://raw.githubusercontent.com/Deepsphere-AI/LVA-Batch5-Assessment/main/Q15_student_grades.csv'
df_student = pd.read_csv(url)
df_student

df_student_ = df_student[df_student['Grade'] != -1]
df_student_


avg_result= df_student_.groupby(['StudentID',	'Subject'], as_index=False).agg({"Grade": "mean"})
avg_result

"""4."""

sensor_data = np.linspace(15, 25, 24)
sensor_data



"""5.

```

import numpy as np
daily_closing_prices = np.array([100, 102, 98, 105, 107, 110, 108, 112, 115, 118, 120])
window_size = 5
```
"""

daily_closing_prices = np.array([100, 102, 98, 105, 107, 110, 108, 112, 115, 118, 120])
window_size = 5

df=pd.DataFrame(daily_closing_prices,columns=['Price'])

df['MA_5'] = df['Price'].rolling(window=5).mean()

print(df)

"""6."""

import numpy as np

mean = [0, 0]
cov_matrix = [[1, 0.5], [0.5, 2]]

samples = np.random.multivariate_normal(mean, cov_matrix, 100)

print(samples)

"""7.

7.
import numpy as np
properties_matrix = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]])
"""

import numpy as np
properties_matrix = np.array([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]])


inverse_matrix = np.linalg.inv(properties_matrix)
inverse_matrix

"""8"""

arr = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])

filtered_arr = arr[np.where(arr > 5)]
print(filtered_arr)

"""9.

```

9.
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace'],
        'Age': [25, 30, 35, 40, 45, 50, 55],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Miami', 'Boston'],
        'Department': ['HR', 'IT', 'Finance', 'Marketing', 'Sales', 'IT', 'HR']}


        ```
"""

import pandas as pd
data = {
    'Name' : ['Alice','Bob','Charlie','David','Eve','Frank','Grace'],
    'Age'  : [25,30,35,40,45,50,55],
    'City' : ['New York','Los Angeles','Chicago','Houston','Pheonix','Miami','Boston'],
    'Department': ['HR','IT','Finance','Marketing','Sales','IT','HR']
}
df = pd.DataFrame(data)
df = df[['Name', 'City']][df['Age'] > 45]
df.set_index('Name', inplace=True)
print(df)

"""10.

```
data = {'Department': ['Electronics', 'Electronics', 'Clothing', 'Clothing', 'Home Goods'],
        'Salesperson': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Sales': [70000, 50000, 30000, 40000, 60000]}

```
"""

data = {'Department':['Electronics','Electronics','Clothing','Clothing','Home goods'],
        'SalesPerson':['Alice','Bob','Charlie','David','Eve'],
        'Sales':[70000,50000,30000,40000,60000]}
df = pd.DataFrame(data)
avg_salesperson = df.groupby(['Department', 'SalesPerson'])['Sales'].mean().reset_index()
avg_department = avg_salesperson.groupby('Department')['Sales'].mean().reset_index()
ranked_departments = avg_department.sort_values(by='Sales', ascending=False)
print(ranked_departments)

"""11.

```

11.
data = {
    'Product': ['Apples', 'Bananas', 'Cherries', 'Dates', 'Elderberries', 'Flour', 'Grapes'],
    'Category': ['Fruit', 'Fruit', 'Fruit', 'Fruit', 'Fruit', 'Bakery', 'Fruit'],
    'Price': [1.20, 0.50, 3.00, 2.50, 4.00, 1.50, 2.00],
    'Promotion': [True, False, True, True, False, True, False]
}

```
"""

import pandas as pd

data = {
    'Product': ['Apples', 'Bananas', 'Cherries', 'Dates', 'Elderberries', 'Flour', 'Grapes'],
    'Category': ['Fruit', 'Fruit', 'Fruit', 'Fruit', 'Fruit', 'Bakery', 'Fruit'],
    'Price': [1.20, 0.50, 3.00, 2.50, 4.00, 1.50, 2.00],
    'Promotion': [True, False, True, True, False, True, False]
}

df = pd.DataFrame(data)

filtered_df = df[(df['Category'] == 'Fruit') & (df['Promotion'] == False) & (df['Price'] > df[df['Category'] == 'Fruit']['Price'].mean())]

print(filtered_df)

"""12.


"""

12.

import pandas as pd
employee_data = {
    'Employee': ['Alice', 'Bob', 'Charlie', 'David'],
    'Department': ['HR', 'IT', 'Finance', 'IT'],
    'Manager': ['John', 'Rachel', 'Emily', 'Rachel']
}

# Dataset of employee project assignments
project_data = {
    'Employee': ['Alice', 'Charlie', 'Eve'],
    'Project': ['P1', 'P3', 'P2']
}
employee_data=pd.DataFrame(employee_data)
project_data=pd.DataFrame(project_data)
merged = employee_data.merge(project_data, on='Employee' ,how='left')
merged.fillna('Not assigned', inplace=True)
print(merged)

"""13."""

import pandas as pd

# Load the dataset of property listings from CSV file
url = 'https://raw.githubusercontent.com/Deepsphere-AI/LVA-Batch5-Assessment/main/Q13_sports_team_stats.csv'
df_sports = pd.read_csv(url)
df_sports
df_sports['Win_ratio']=[df_sports['Wins']/df_sports['GamesPlayed']]
df_sports

"""14"""

import pandas as pd

# Load the dataset of property listings from CSV file
url = 'https://raw.githubusercontent.com/Deepsphere-AI/LVA-Batch5-Assessment/main/Q14_customer_purchases.csv'
df_purchase = pd.read_csv(url)
df_purchase
df_purchase['Date'] = pd.to_datetime(df_purchase['Date'])
df_purchase['Month'] = df_purchase['Date'].dt.month
monthly_revenue = df_purchase.groupby('Month')['PurchaseAmount'].sum()
monthly_revenue

"""15."""

import pandas as pd

# Load the dataset of property listings from CSV file
url = 'https://raw.githubusercontent.com/Deepsphere-AI/LVA-Batch5-Assessment/main/Q15_student_grades.csv'
df_student = pd.read_csv(url)
df_student


avg_result= df_student_.groupby(['Subject'], as_index=False).agg({"Grade": "mean"})
avg_result






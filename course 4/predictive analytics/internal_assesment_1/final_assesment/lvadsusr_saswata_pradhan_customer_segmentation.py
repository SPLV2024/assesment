# -*- coding: utf-8 -*-
"""LVADSUSR_SASWATA_Pradhan_Customer_segmentation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1O5385fqG2KTt07EFQxiGwlVjTLhapJbD
"""

#Importing the Libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)

url= 'https://raw.githubusercontent.com/Deepsphere-AI/LVA-Batch5-Assessment/main/customer_segmentation.csv'

#Loading the dataset
data = pd.read_csv(url,sep=',')
print("Number of datapoints:", len(data))
data.head()

data.shape

data.columns

data.info()

sns.heatmap(data.isnull(),cmap = 'magma',cbar = False);

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
df_imputed = impute_with_custom(data)

# Check if there are any missing values left
print(df_imputed.isnull().sum())

fig,ax = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5))

plt.subplot(1,1,1)
sns.heatmap(data.describe().T[['mean']],cmap = 'inferno_r',annot = True,fmt = '.2f',linecolor = 'black',linewidths = 0.4,cbar = False);
plt.title('Mean Values');

fig.tight_layout(pad = 3)

data.drop(['ID','Dt_Customer'],axis=1,inplace=True)

col = list(data.columns)
categorical_features = []
numerical_features = []
for i in col:
    if len(data[i].unique()) > 6:
        numerical_features.append(i)
    else:
        categorical_features.append(i)

print('Categorical Features :',*categorical_features)
print('Numerical Features :',*numerical_features)

print("Total categories in the feature Marital_Status:\n", data["Marital_Status"].value_counts(), "\n")
print("Total categories in the feature Education:\n", data["Education"].value_counts())

#Feature Engineering
#Age of customer today
data["Age"] = 2021-data["Year_Birth"]

#Total spendings on various items
data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]

#Deriving living situation by marital status"Alone"
data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})

#Feature indicating total children living in the household
data["Children"]=data["Kidhome"]+data["Teenhome"]

#Feature for total members in the householde
data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2})+ data["Children"]

#Feature pertaining parenthood
data["Is_Parent"] = np.where(data.Children> 0, 1, 0)

#Segmenting education levels in three groups
data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

#For clarity
data=data.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})

import pandas as pd
import numpy as np

# Define a function to remove outliers using Z-score
def remove_outliers_zscore(data, threshold=3):
    z_scores = (data - data.mean()) / data.std()
    filtered_data = data[(np.abs(z_scores) < threshold).all(axis=1)]
    return filtered_data

# Assuming 'data' is your DataFrame with numeric columns
# Apply Z-score outlier removal for numeric columns
numeric_data = data.select_dtypes(include=np.number)
filtered_data = remove_outliers_zscore(numeric_data)

print("The total number of data-points after removing the outliers is:", len(filtered_data))

# Exclude non-numeric columns
numeric_data = data.select_dtypes(include=np.number)

# Calculate correlation matrix
corrmat = numeric_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(20, 20))
sns.heatmap(corrmat, annot=True, cmap='coolwarm', center=0)
plt.show()

def select_features(data, threshold=0.5):
    # Exclude non-numeric columns
    numeric_data = data.select_dtypes(include=np.number)

    # Calculate correlation matrix
    corrmat = numeric_data.corr()

    # Filter features based on correlation threshold
    selected_features = set()
    for i in range(len(corrmat.columns)):
        for j in range(i):
            if abs(corrmat.iloc[i, j]) > threshold:
                col_i = corrmat.columns[i]
                col_j = corrmat.columns[j]
                # Add both features to the selected features set
                selected_features.add(col_i)
                selected_features.add(col_j)

    return selected_features

# Automatically select features based on a correlation threshold
selected_features = select_features(data, threshold=0.5)
print("Selected features based on correlation threshold:", selected_features)

#Get list of categorical variables
s = (data.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)

#Label Encoding the object dtypes.
LE=LabelEncoder()
for i in object_cols:
    data[i]=data[[i]].apply(LE.fit_transform)

print("All features are now numerical")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Define X (features)
X = data.copy()  # Assuming all columns are features

# Standardize the features
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=3)
pca.fit(scaled_X)

# Determine the number of components to keep based on explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
num_components = np.argmax(cumulative_explained_variance_ratio >= 0.95) + 1  # Keep components explaining at least 95% of variance

# Fit PCA with the selected number of components
pca = PCA(n_components=num_components)
selected_features = pca.fit_transform(scaled_X)

# Create a DataFrame with the transformed features
transformed_df = pd.DataFrame(selected_features, columns=[f'PC{i}' for i in range(1, num_components + 1)])

print("Transformed features after PCA:")
transformed_df.head()

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

# Instantiate the KElbowVisualizer with a KMeans model
elbow_visualizer = KElbowVisualizer(KMeans(), k=(2, 10))  # Try different numbers of clusters (2 to 10 in this case)

# Fit the data to the visualizer
elbow_visualizer.fit(transformed_df)

# Show the plot
elbow_visualizer.show()

from sklearn.cluster import KMeans

# Get the optimal number of clusters from the elbow visualizer
optimal_n_clusters = elbow_visualizer.elbow_value_

# Instantiate the KMeans model with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)

# Fit the KMeans model to the data
kmeans.fit(transformed_df)

# Get cluster labels for each data point
cluster_labels = kmeans.labels_

# Print the cluster centers (centroids)
print("Cluster centers:")
print(kmeans.cluster_centers_)

# Print the labels assigned to each data point
print("Cluster labels for each data point:")
print(cluster_labels)

from sklearn.metrics import silhouette_score

# Calculate the silhouette score
silhouette_avg = silhouette_score(transformed_df, cluster_labels)
print(f'Silhouette Score: {silhouette_avg}')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""#1. Load Data"""

df = pd.read_csv('/content/Credit Card Customer Data.csv')



"""#2. Data Pre-processing - handle missing values, outliers, etc.
Dropping unnecessary columns

Handling missing values

Applying log transformation to handle outliers
"""

# Drop 'Customer Key' and 'Sl_No' as they are unique identifiers
df.drop(['Customer Key', 'Sl_No'], axis=1, inplace=True)

# Drop rows where 'Avg_Credit_Limit' is missing
df.dropna(subset=['Avg_Credit_Limit'], inplace=True)

# Apply log transformation to reduce the impact of outliers
cols = ['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_bank', 'Total_visits_online', 'Total_calls_made']
for col in cols:
    df[col] = np.log(1 + df[col])

# Fill missing values with the median of each column
df = df.fillna(df.median())

"""#3. Exploratory Data Analysis - descriptive stats, shape of the data, etc.

Displaying the first few rows, summary statistics, information about

the dataset, and checking the percentage of missing values.

Visualizing the distribution of the transformed columns.
"""

# Display the first few rows of the dataset
print(df.head())

# Display summary statistics of the dataset
print(df.describe())

# Display information about the dataset
print(df.info())

# Check the percentage of missing values in each column
print(df.isna().mean() * 100)

# Visualize the distribution of the transformed columns
plt.figure(figsize=(15, 20))
for i, col in enumerate(cols):
    ax = plt.subplot(6, 2, i+1)
    sns.kdeplot(df[col], ax=ax)
plt.show()

# Displaying the shape and descriptive statistics again after processing
print(df.describe())
print(df.shape)

df = df.fillna(df.median())

"""#4. Model Training & Testing
Performing PCA for dimensionality reduction

Using the elbow method and silhouette scores to determine the optimal number of clusters

Implementing KMeans clustering with the optimal number of clusters
"""

# Perform PCA to reduce dimensionality while retaining 95% of the variance
pca = PCA(n_components=0.95)
X_red = pca.fit_transform(df)

# Determine the optimal number of clusters using the elbow method
kmeans_models = [KMeans(n_clusters=k, random_state=23).fit(X_red) for k in range(1, 10)]
inertia = [model.inertia_ for model in kmeans_models]

plt.plot(range(1, 10), inertia)
plt.title('Elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Evaluate the number of clusters using silhouette scores
silhouette_scores = [silhouette_score(X_red, model.labels_) for model in kmeans_models[1:4]]
plt.plot(range(2, 5), silhouette_scores, "bo-")
plt.xticks([2, 3, 4])
plt.title('Silhouette scores vs Number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.show()

# Implement KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=3, random_state=23)
kmeans.fit(X_red)

"""#5. Model evaluation metrics
Calculating the silhouette score for the model
"""

# Calculate the silhouette score for evaluation
print('Silhouette score of our model is ' + str(silhouette_score(X_red, kmeans.labels_)))

"""#6. Business recommendations

Assigning cluster labels and visualizing clusters
Providing detailed business recommendations based on the clustering analysis
"""

# Assign cluster labels to the original data
df['cluster_id'] = kmeans.labels_

# Reverse the log transformation for visualization
for col in cols:
    df[col] = np.exp(df[col]) - 1

# Visualize the clusters based on key features
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Total_visits_online', y='Total_calls_made', hue='cluster_id')
plt.title('Distribution of clusters based on Total visits online and Total calls made')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Avg_Credit_Limit', y='Total_calls_made', hue='cluster_id')
plt.title('Distribution of clusters based on Avg Credit Limit and Total calls made')
plt.show()

"""# Business Recommendations

"""

# ### Business Recommendations:

# 1. Customer Segmentation:
#    - Based on the clustering analysis, segment customers into distinct groups. Each cluster represents a group of customers with similar behaviors and credit card usage patterns.
#    - Develop targeted marketing strategies for each customer segment to increase engagement and retention.

# 2. Credit Limit Management:
#    - Analyze the average credit limit within each cluster. Offer tailored credit limit increases to segments with higher credit utilization and good repayment history to encourage higher spending and increase interest income.

# 3. Product Customization:
#    - Customize credit card products and services based on the characteristics of each segment. For example, offer premium cards with exclusive benefits to high-spending clusters and basic cards with lower fees to cost-sensitive clusters.

# 4. Customer Service Enhancement:
#    - Identify clusters with higher frequencies of bank visits, online interactions, and customer service calls. Provide personalized service options, such as dedicated account managers for high-value segments or improved online support for tech-savvy customers.

# 5. Fraud Detection and Risk Management:
#    - Use clustering results to enhance fraud detection algorithms. Monitor unusual activities within clusters and implement stricter controls for segments with higher risk profiles.

# 6. Customer Retention Strategies:
#    - Develop retention programs for clusters with high churn risk. Offer incentives like cashback, loyalty points, or lower interest rates to retain valuable customers.

# 7. Cross-Selling Opportunities:
#    - Identify potential cross-selling opportunities within each cluster. For example, offer personal loans, insurance products, or investment services to segments showing interest in multiple financial products.


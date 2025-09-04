#Selina Hui
#ITP259 Fall 2024
#HW 2

# Question 1: Reading dataset into a dataframe & importing the header
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Suppressing the printing of dataframe metadata
pd.set_option("display.max_columns", None)
pd.set_option('display.show_dimensions', False)

# Reading data set into a data frame
wineData = pd.read_csv("/Users/Selina/Documents/ITP259/Lecture5_UnsupervisedLearning/wineQualityReds.csv")
wineDataFrame = pd.DataFrame(wineData)

# Printing of imported header
print("Imported header: ")
print(wineDataFrame.head())
print("\n")

# Question 2: Drop Wine column from the data frame
wineDataFrame.drop('Wine', axis=1, inplace=True)
print("Dropped data frame: ")
print(wineDataFrame.head())
print("\n")

# Question 3: Extract Quality and store it in a separate variable
wine_quality = wineDataFrame['quality']

# Question 4: Drop Quality column from the original data frame
wineDataFrame = wineDataFrame.drop('quality', axis=1)

# Question 5: Print wine data frame and wine quality data frame.
print("Wine data frame: ")
print(wineDataFrame.head())
print("\n")

print("Wine quality new data frame: ")
print(wine_quality.head())
print("\n")

# Question 6: Normalize all columns of data frame using Normalizer class
norm = Normalizer()
norm.fit(wineDataFrame)
wineData_norm = pd.DataFrame(norm.transform(wineDataFrame), columns=wineDataFrame.columns)

# Question 7: Print normalized data frame
print("Normalized columns of wine data: ")
print(wineData_norm)
print("\n")

# Question 8: Create a range of k-values (1:11) for k-means clustering.
ks = range(1,11)

# Iterate on the k-values and store the inertia for each clustering in a list
inertia = []
for k in ks:
    wineKMeans = KMeans(n_clusters=k, random_state=0)
    wineKMeans.fit(wineData_norm)
    inertia.append(wineKMeans.inertia_)

print("Inertia clusters list: ")
print(inertia)
print("\n")

# Question 9: Plot the chart of inertia vs. k, the number of clusters
plt.plot(ks, inertia, "b-o")
plt.xlabel("Number of clusters, k")
plt.ylabel("Inertia")
plt.title('Elbow method for Optimal K-means Clustering')
plt.xticks(ks)
plt.show()

# Question 11: Instantiate the k-means model
wineKMeans = KMeans(n_clusters=6, random_state=2023)
wineKMeans.fit(wineData_norm)

# Assign respective cluster number to each wine & print data frame.
wineData_norm['cluster'] = wineKMeans.labels_
print("Wine data frame with cluster numbers: ")
print(wineData_norm)
print("\n")

# Question 12: Add the 'quality' data back into the data frame
wineData_norm['quality'] = wine_quality
print("Wine data frame with quality column added: ")
print(wineData_norm.head())
print("\n")

# Question 13: Create a crosstab of cluster number vs. quality
wineCrossTab = pd.crosstab(wineData_norm['quality'], wineData_norm['cluster'])
print(wineCrossTab)

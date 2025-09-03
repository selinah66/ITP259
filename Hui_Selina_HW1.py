#Selina Hui
#ITP259 Fall 2024
#HW 1

# Question 1: Reading dataset into a dataframe & importing the header
import pandas as pd

# Suppressing the printing of dataframe metadata
pd.set_option('display.show_dimensions', False)

# Reading data set into a data frame
data = pd.read_csv('/Users/Selina/Documents/ITP259/Lecture5_UnsupervisedLearning/wineQualityReds.csv')
frame = pd.DataFrame(data)

# Printing of imported header
print("Imported header of data set: ")
print(frame.head())
print("\n")

# Question 2: Print the first 10 rows of the data frame
print("First 10 rows of the data frame: ")
print(frame.head(10))
print("\n")

# Question 3: Print the data frame in descending order of volatility
print("Wine Data in Descending Order of Volatility: ")
sorted_frame = frame.sort_values(by='volatile.acidity', ascending=False)
print(sorted_frame)
print("\n")

# Question 4: Display all the wines that have a quality of 7
print("All wines with quality of 7: ")
print(frame.loc[frame['quality'] == 7, [frame.columns[0], 'quality']])
print("\n")

# Question 5: Calculates the average pH of all wines
print("Average pH of all wines:")
print(frame['pH'].mean())
print("\n")

# Question 6: Calculate and count the number of wines with alcohol level over 10
print("Number of wines with alcohol level above 10:")
print((frame['alcohol'] > 10).sum())
print("\n")

# Question 7: Find the wine with the highest alcohol level
print("Wine with highest alcohol level: ")
max_row = frame[frame['alcohol'] == frame['alcohol'].max()]
print(frame.loc[frame['alcohol'] == frame['alcohol'].max(), [frame.columns[0], 'alcohol']])
print("\n")

# Question 8: List the residual sugar level of a random wine and suppresses display of extra data frame metadata
import random

random_num = random.randint(0, len(frame) - 1)
random_wine = frame.iloc[random_num]
print(f"Residual sugar level of a random wine - {random_num}:")
print(random_wine[['residual.sugar']].to_string())
print("\n")

# Question 9: List a random wine with quality of 4
wine_quality4 = frame.loc[frame['quality'] == 4]
num_wine4 = len(wine_quality4)
random_quality4 = random.randint(0, num_wine4 - 1)
random_quality_wine = wine_quality4.iloc[random_quality4]
print(f"Random quality-4 wine - index {random_num}:")
print(random_quality_wine[['quality']].to_string())
print("\n")

# Question 10: Drop wines with quality 4 and count number of wines left
wine_not_quality4 = frame[frame['quality'] != 4]
print("Dataframe without wines of quality 4: ")
print(wine_not_quality4)
print("\n")

print("Number of wines without a quality of 4: ")
print(len(wine_not_quality4))
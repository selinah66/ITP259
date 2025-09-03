#Selina Hui
#ITP259 Fall 2024
#HW 3 Problem #1: Logistic Regression

# Question 1: Reading dataset into a dataframe.
import pandas as pd
import numpy as np

# Reading data set into a data frame
TitanicData = pd.read_csv("/Users/Selina/Documents/ITP259/Lecture6_Linear&LogisticRegression/Titanic.csv")
pd.set_option("display.max_columns", None)
TitanicDataFrame = pd.DataFrame(TitanicData)

print("Titanic data: ")
print(TitanicDataFrame.head())

# Question 2: Explore the dataset and determine what is the target variable.
print("\nThe target variable is survival, as we are exploring what factors affect survivability of the Titanic passengers.")

# Question 3: Drop factor(s) that are not likely to be relevant for logistic regression.
# I dropped the irrelevant factors of passenger number as it is irrelevant in predicting survival.
TitanicDataFrame.drop('Passenger', axis=1, inplace=True)

print("\nTitanic data frame after dropping: ")
print(TitanicDataFrame.head())

# Question 4: Convert all categorical feature variables into dummy variables.
TitanicDataFrame = pd.get_dummies(TitanicDataFrame, columns=['Class'], prefix='Class', dtype=int) # for class
TitanicDataFrame = pd.get_dummies(TitanicDataFrame, columns=['Sex'], prefix='Sex', dtype=int) # for sex
TitanicDataFrame = pd.get_dummies(TitanicDataFrame, columns=['Age'], prefix='Age', dtype=int) # for age
TitanicDataFrame['Survived'] = TitanicDataFrame['Survived'].map({'No': 0, 'Yes': 1}) # for survival

print("\nTitanic data frame with dummy variables: ")
print(TitanicDataFrame.head())

# Question 5: Assign X and y.
X = TitanicDataFrame.drop('Survived', axis=1)
y = TitanicDataFrame['Survived']

print(X.head())
print("\nShape of X: ", X.shape)
print(y.head())
print("\nShape of y:", y.shape)

# Question 6: Partition the data into train and test sets (70/30) using random_state = 2023, stratify the data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2023)

# Print shape of training and test set
print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Print distribution of target variable in training and test sets
print("\nTraining set survival distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest set survival distribution:")
print(y_test.value_counts(normalize=True))

# Question 7: Fit the training data to a logistic regression model.
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
logReg.fit(X_train, y_train)

print("\nModel score on training data:", logReg.score(X_train, y_train))
print("Model score on testing data:", logReg.score(X_test, y_test))

# Question 8: Display the accuracy of your predictions.
from sklearn.metrics import accuracy_score
y_pred = logReg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy: {:.2f}".format(accuracy))

# Question 9: Plot the lift curve.
import matplotlib.pyplot as plt

# Generate predicted probabilities for lift curve.
y_proba = logReg.predict_proba(X_test)[:, 1]

# Calculate cumulative gains and the lift.
y_true_sorted = [y for _, y in sorted(zip(y_pred, y_test), reverse=True)]
cumulative_gains = np.cumsum(y_true_sorted)
lift = cumulative_gains / np.sum(y_test)

# Plot the lift curve.
plt.figure(figsize=(9, 8))
plt.plot(np.linspace(0, 1, len(lift)), lift, label='Logistic Regression Model')
plt.xlabel('Percentage of sample')
plt.ylabel('Percentage of positive outcomes')
plt.title('Lift Curve of Titanic Data')
plt.grid(True)
plt.show()

# Question 10: Plot the confusion matrix along with the labels of Yes and No.
# Create the confusion matrix.
from sklearn.metrics import confusion_matrix
conf_matr = confusion_matrix(y_test, y_pred)

# Plot confusion matrix using seaborn.
import seaborn as sns
class_labels = ['No', 'Yes']
plt.figure(figsize=(9, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)

# Label and plot confusion matrix.
plt.xlabel('Predicted Survival')
plt.ylabel('Actual Survival')
plt.title('Confusion Matrix for Titanic Survival Prediction')
plt.show()

# Question 11: Display the predicted value of the survivability of a male adult passenger traveling in 3rd class
# Define passenger data for male adult passenger traveling in 3rd class.
passengerData = {
    'Class_1st': 0,
    'Class_2nd': 0,
    'Class_3rd': 1,
    'Class_Crew': 0,
    'Sex_Female': 0,
    'Sex_Male': 1,
    'Age_Adult': 1,
    'Age_Child': 0
}

# Place passenger data into a data frame for prediction using logistic regression.
examplePassenger = pd.DataFrame([passengerData])

# Predict and print predicted survival and probability of survival of the passenger.
passengerPrediction = logReg.predict(examplePassenger)
passengerProbability = logReg.predict_proba(examplePassenger)

print(f"\nPredicted survival: {'Yes' if passengerPrediction[0] == 1 else 'No'}")
print(f"\nProbability of survival: {passengerProbability[0][1]:.2f}")
#Selina Hui
#ITP259 Fall 2024
#HW 3 Problem #2: k-Nearest Neighbors

# Question 1: Create a DataFrame to store the diabetes data
import pandas as pd
import matplotlib.pyplot as plt

# Reading data set into a data frame
DiabetesData = pd.read_csv("/Users/Selina/Documents/ITP259/Lecture6_Linear&LogisticRegression/diabetes.csv")
pd.set_option("display.max_columns", None)
diabetes_knn = pd.DataFrame(DiabetesData)

# Print diabetes data
print("Diabetes data: ")
print(diabetes_knn)

# Question 2: Create the Feature Matrix (X) and Target Vector (y)
X = diabetes_knn.drop('Outcome', axis=1)
# alternative: X=diabetes_knn.iloc[:,:-1]
print(X.shape)
y = diabetes_knn['Outcome']
print(y.shape)

# Question 3: Standardize the attributes of the Feature Matrix (x)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Print the first few rows after scaling the feature matrix
print("\nFirst few rows of scaled features:")
print(X_scaled[:5])

# Question 4: Split the Feature Matrix and Target Vector into three partitions.
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2023)
X_trainA, X_trainB, y_trainA, y_trainB = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=2023)

# Print the shapes of the three partitions of the feature matrix and the target vector
print("\nShapes after splitting:")
print("X_trainA:", X_trainA.shape)
print("X_trainB:", X_trainB.shape)
print("X_test:", X_test.shape)
print("y_trainA:", y_trainA.shape)
print("y_trainB:", y_trainB.shape)
print("y_test:", y_test.shape)

# Question 5: Develop a KNN based model based on Training A for various ks. K should range between 1 and 30
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
ks = list(range(1, 31))

# Print list of all k's as evidence.
print("\nk's: ", ks)

# Create lists for scores A and B
scoresA = []
scoresB = []

# For-loop for developing KNN model and computing the KNN accuracy score for training datasets.
for k in ks:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_trainA, y_trainA)
    y_pred_A = model.predict(X_trainA)
    scoresA.append(accuracy_score(y_trainA, y_pred_A))
    y_pred_B = model.predict(X_trainB)
    scoresB.append(accuracy_score(y_trainB, y_pred_B))

# Question 6: Compute the KNN score (accuracy) for training A and training B data
# Display accuracy values for training A and Bx
for k, score_A, score_B in zip(ks, scoresA, scoresB):
    print(f"\nk={k}: Training A accuracy = {score_A:.4f}, Training B accuracy = {score_B:.4f}")

# Question 7: Plot a graph of training A and training B accuracy.
plt.figure(figsize=(10, 6))
plt.plot(ks, scoresA, label='Training Set A')
plt.plot(ks, scoresB, label='Training Set B')
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.legend()
plt.show()

# Question 8: Using the selected value of k, score the test data set.
optimal_k = 15
knn_model = KNeighborsClassifier(n_neighbors=optimal_k)
knn_model.fit(X_trainB, y_trainB)
y_pred = knn_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest data accuracy with k={optimal_k}: {test_accuracy:.4f}")

# Question 9: Plot the confusion matrix as a figure.
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Create the confusion matrix for the diabetes data
conf_matr = confusion_matrix(y_test, y_pred)
class_labels = ['No', 'Yes']
plt.figure(figsize=(9, 8))
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)

# Label and plot confusion matrix.
plt.xlabel('Predicted Diabetes Diagnosis')
plt.ylabel('Diabetes Diagnosis')
plt.title('Confusion Matrix for Diabetes Diagnosis Prediction')
plt.show()

# Question 10: Predict the Outcome for a person with 2 pregnancies, 150 glucose, 85 blood pressure, 22 skin thickness, 200 insulin, 30 BMI, 0.3 diabetes pedigree, 55 age.
patientData = {
    'Pregnancies': 2,
    'Glucose': 150,
    'BloodPressure': 85,
    'SkinThickness': 22,
    'Insulin': 200,
    'BMI': 30,
    'DiabetesPedigreeFunction': 0.3,
    'Age': 55
}
# patient = [[2, 150, 85, 22....]]

# Place patient data into a data frame for prediction using k-nearest neighbors
examplePatient = pd.DataFrame([patientData])

# Predict and print predicted diabetes and probability of the patient have diabetes.
patientPrediction = knn_model.predict(examplePatient)
patientProbability = knn_model.predict_proba(examplePatient)
# pd.DF(standard.transform(patient), columns=X.columns)

print(f"\nPredicted diabetes: {'Yes' if patientPrediction[0] == 1 else 'No'}")
print(f"\nProbability of diabetes: {patientProbability[0][1]:.2f}")

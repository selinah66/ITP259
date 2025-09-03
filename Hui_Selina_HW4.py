#Selina Hui
#ITP259 Fall 2024
#HW 4: MLPClassifier for Spiral Blobs

# Question 2.1: Generate x,y coordinates of spirally distributed blobs in two colors.
import pandas as pd
import numpy as np

# Create function for generating spiral data.
def make_spiral_blob(n_samples, noise):
    theta = np.linspace(0, 2 * np.pi, n_samples)

    # First spiral
    r1 = (2 * theta) + np.pi
    x1 = r1 * np.cos(theta) + np.random.normal(0, noise, n_samples)
    y1 = r1 * np.sin(theta) + np.random.normal(0, noise, n_samples)

    # Second spiral
    r2 = (-2 * theta) - np.pi
    x2 = r2 * np.cos(theta) + np.random.normal(0, noise, n_samples)
    y2 = r2 * np.sin(theta) + np.random.normal(0, noise, n_samples)

    # Combine the spirals
    x_coord = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2))))
    y_coord = np.hstack((np.zeros(n_samples), np.ones(n_samples)))

    return x_coord, y_coord

# Generate the spiral data using function.
X, y = make_spiral_blob(1000, 0.5)

# Question 2.2: Display a scatter plot of the x and y-coordinates using the label as color.
# Plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor= 'black', s=30)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Spiral Data')
plt.show()

# Question 2.3: Create partitions with 70% train dataset, stratify the split with random state 2023.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2023)

# Print the shapes of the partitions
print("\nShapes after splitting:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Create MLPClassifier Neural Network.
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50, 50), activation='relu', alpha=0.0001, solver='adam', random_state=2023, learning_rate_init=0.01, max_iter=1000, verbose=True)

# Fit MLPClassifier to training data.
mlp.fit(X_train, y_train)

# Plot loss curve
plt.figure()
plt.plot(mlp.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Cross-entropy loss")
plt.title('MLPClassifier Loss Curve')
plt.show()

# Accuracy of model on test data
print("\nModel accuracy on Test Data:", mlp.score(X_test,y_test))

# Create and plot confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = mlp.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=mlp.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_).plot()
plt.title('Confusion Matrix for MLPClassifier')
plt.show()

# Question 2.8: Creating the decision boundary.

# Mesh of X and y-coordinates
X1 = np.arange(-20, 20, 0.1)
X2 = np.arange(-20, 20, 0.1)

X1, X2 = np.meshgrid(X1, X2)

# Reshape Meshgrid to Dataframe
X_flat = np.c_[X1.ravel(), X2.ravel()]
df = pd.DataFrame(X_flat, columns=['X1', 'X2'])

# Classify each point using trained model
Z = mlp.predict(X_flat)

# Reshape to match meshgrid shape
Z = Z.reshape(X1.shape)

# Plot of Spiral Data and Mesh Data
plt.figure(figsize=(10, 8))
contour = plt.contourf(X1, X2, Z, alpha=0.8, cmap='coolwarm')

plt.scatter(X[:, 0], X[:,1], c=y, cmap='coolwarm', edgecolor= 'black', s=20)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('MLPClassifier Decision Boundary for Spiral Data')
plt.xlim(-20, 20)
plt.ylim(-20, 20)

plt.colorbar(contour)
plt.show()
#Selina Hui
#ITP259 Fall 2024
#HW 5: Digit Classifier

# Create a DataFrame to store the data
import pandas as pd
import numpy as np

# Question 1: Reading data set into a data frame
HandwrittenDF = pd.read_csv("/Users/Selina/Documents/ITP259/Lecture9_DigitClassification/A_Z Handwritten Data.csv")
pd.set_option("display.max_columns", None)
digits = pd.DataFrame(HandwrittenDF)
print("First 5 rows of digit data: ", digits.head())

# Question 3: Separate the dataframe into feature set and target variable
X = HandwrittenDF.iloc[:,1:]
y = HandwrittenDF.iloc[:,0]

# Question 4: Print feature set and target variable.
print("\nFeature set: ", X.shape)
print("Target variable: ", y.shape)

# Question 6: Mapping numbers to letters.
import string

word_dict = dict(enumerate(string.ascii_lowercase))
print("\nLetter mapping:\n", word_dict)

# Question 7: Histogram count of the letters
import matplotlib.pyplot as plt
import seaborn as sb
plt.figure(1, figsize=(10, 6))
ax = sb.countplot(x="label", data=HandwrittenDF)
ax.set_xticks(list(word_dict.keys()))
ax.set_xticklabels(word_dict.values())
plt.title('Histogram of Letters')
plt.xlabel('Letter')
plt.ylabel('Frequency')

# Label x-axis with letters
plt.xticks(range(26), [chr(65+i) for i in range(26)])
plt.grid(True, alpha=0.3)
plt.show()

# Question 8: Display 64 random letters from the dataset with labels.
random_index = np.random.choice(len(X), 64, replace=False)

X_samples = X.iloc[random_index].to_numpy()
y_labels = y.iloc[random_index].to_numpy()

plt.figure(1)
fig, axs = plt.subplots(8, 8, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    img = X_samples[i].reshape(28, 28)
    ax.set_title(f'Label: {word_dict[y_labels[i]]}')
    ax.axis('off')
    ax.imshow(img, cmap='gray')
plt.grid(False)
plt.tight_layout()
plt.show()

# Question 9: Partition the data into train and test sets (70/30). Use random_state = 2023. Stratify it.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2023)
print("\nTrain/test shapes: ", X_train.shape, X_test.shape)

# Question 10: Scale the train and test features.
X_train = X_train/255.0
X_test = X_test/255.0

# Question 11: Create an MLP Classifier.
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50, 50), activation="relu",
                    max_iter=20, alpha=1e-3, solver="adam",
                    random_state=2023, learning_rate_init=0.01, verbose=True)

# Question 12: Fit to train the model
mlp.fit(X_train, Y_train)

# Question 13: Plot the loss curve.
plt.plot(mlp.loss_curve_)
plt.title('Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

# Question 14: Display model accuracy
print("\nModel Accuracy: ", mlp.score(X_test,Y_test))

# Question 15: Plot confusion matrix along with the letters.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = mlp.predict(X_test)
cm = confusion_matrix(y_pred, Y_test)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.figure(figsize= (10,10))
plt.show()

# Question 16: Predict and display both actual and predicted letters of first row of test dataset.
first_row = X_test.iloc[0,:]
first_row = np.array(first_row)
first_row = first_row.reshape(28, 28)

predicted_label = word_dict[y_pred[0]]
actual_label = word_dict[Y_test.iloc[0]]

plt.title(f"The predicted letter is: {predicted_label}, the actual letter is: {actual_label}")
plt.imshow(first_row, cmap='gray')  # expects an ARRAY
plt.show()

# Question 17: Display the actual and predicted letter of a misclassified letter.
# Filter test data to prediction failure with new dataframe
failed_df = X_test[y_pred != Y_test]

failed_index = failed_df.sample(n=1).index[0]

req_id = Y_test.index.get_loc(failed_index) # trick to align the original DF and the new DF

failed_row = np.array(X_test.iloc[req_id]).reshape(28, 28)
plt.imshow(failed_row, cmap='gray') # expects an ARRAY

actual_failed_labels = word_dict[Y_test.iloc[req_id]]
predicted_failed_labels = word_dict[y_pred[req_id]]

plt.title(f"Prediction of failed letter: {predicted_failed_labels}, the actual failed letter: {actual_failed_labels}")
plt.axis('off')
plt.show()

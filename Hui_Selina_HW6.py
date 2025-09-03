#Selina Hui
#ITP259 Fall 2024
#HW 6

import keras
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

# Question 1: Load the dataset from Keras.
fashion = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()

# Combine test and training data into a single array for images and labels, flattened.
all_images = np.vstack((train_images.reshape(train_images.shape[0], -1),
                        test_images.reshape(test_images.shape[0], -1)))
all_labels = np.concatenate((train_labels, test_labels))

# Create column names for each pixel and the label
image_columns = [f'pixel_{i}' for i in range(784)]  # 28x28 = 784 pixels
columns = image_columns + ['label']

# Create dataframe and save as CSV file.
fashion_df = pd.DataFrame(np.column_stack((all_images, all_labels)), columns=columns)
fashion_df.to_csv('fashion.csv', index=False)
print(fashion_df.head())

# Question 2: Separate dataset into feature set and target variable.
x = fashion_df.drop("label", axis=1)
y = fashion_df["label"]

print(x.head())
print(y.head())

# Separate test and training partitions.
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2023)

# Question 3: Print the shapes of the train and test sets for the features and target.
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Question 5: Map numbers to clothing using a data dictionary.
clothing_dict = {0: "T-shift/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}

# Question 6: Show a histogram count of the apparel.
plt.figure(1, figsize=(10, 6))
ax = sb.countplot(x="label", data=fashion_df)
ax.set_xticks(list(clothing_dict.keys()))
ax.set_xticklabels(clothing_dict.values())

plt.title('Histogram of Clothing')
plt.xlabel('Letter')
plt.ylabel('Count')
plt.show()

# Question 7: Display 25 random apparel from the train dataset and their labels.
random_index = np.random.choice(len(X_train), 25, replace=False)

X_samples = X_train.iloc[random_index].to_numpy()
y_labels = y_train.iloc[random_index].to_numpy()

plt.figure(1)
fig, axs = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    img = X_samples[i].reshape(28, 28)
    ax.set_title(f'{clothing_dict.get(int(y_labels[i]))}')
    ax.axis('off')
    ax.imshow(img, cmap='gray')
plt.grid(False)
plt.tight_layout()
plt.show()

# Question 8: Scale the train and test features.
X_train = X_train/255.0
X_test = X_test/255.0

# Reshape training and testing data
X_train = X_train.values.reshape(-1, 28, 28)
X_test = X_test.values.reshape(-1, 28, 28)

# Question 9: Create a keras model of sequence of layers.
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(28, 28)))
model.add(keras.layers.Flatten()) # must flatten input first
model.add(keras.layers.Dense(50, activation = "relu"))
model.add(keras.layers.Dense(50, activation = "relu"))

# Question 10: Add a dense layer as output layer.
model.add(keras.layers.Dense(10, activation = "softmax"))

# Question 11: Display model summary.
model.summary()

# Question 12: Set the model loss function, optimizer, and metrics.
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "sgd", metrics = ["accuracy"])

# Question 13: Fit to train the model with at least 100 epochs.
h = model.fit(X_train, y_train, epochs = 100, verbose=1)

# Question 14: Plot the loss curve
loss = h.history['loss']
accuracy = h.history['accuracy']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(10, 10))
plt.plot(loss, label='Loss', color='tab:blue')
plt.plot(accuracy, label='Accuracy', color='tab:orange')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Model Loss and Accuracy')
plt.legend()
fig.tight_layout()
plt.show()

# Question 15: Display the accuracy of the model
test_loss, test_accuracy = model.evaluate(X_test,y_test)
print(f'Test Accuracy: {test_accuracy}')

# Question 16: Display the predicted apparel of the first row in the test dataset. Also display the actual apparel. Show both actual and predicted letters (as title) on the image of the apparel.
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Predicted apparel from first row
image_number = random.randint(0, len(X_test) - 1)
test_sample = X_test[image_number].reshape(28, 28)

predicted_label = int(y_pred[image_number])
actual_label = int(y_test.iloc[image_number])

predicted_apparel = clothing_dict.get(predicted_label, f'({predicted_label}')
actual_apparel = clothing_dict.get(actual_label, f'({actual_label}')

# Plot the predicted apparel with labels.
plt.figure(figsize=(8, 6))
plt.imshow(test_sample, cmap="gray")
plt.title(f"The Predicted Apparel is {predicted_apparel} \nand the Actual Apparel is {actual_apparel}")
plt.axis('off')
plt.show()

# Question 17: Failed prediction and actual apparel value
# Filter the test dataframe to those cases where the prediction failed
failed_apparel = np.where(y_pred != y_test)[0]
print("Incorrect apparel predictions ", failed_apparel)

# Pick a random row index from the failed dataframe
failed_index = random.choice(failed_apparel)
print("The index of the row of a random incorrect prediction ", failed_index)

# Unflatten the row at the failed index.
failed_sample = X_test[failed_index].reshape(28, 28)

predicted_failed_apparel = int(y_pred[failed_index])
actual_failed_apparel = int(y_test.iloc[failed_index])

predicted_failed_apparel_label = clothing_dict.get(predicted_failed_apparel)
actual_failed_apparel_label = clothing_dict.get(actual_failed_apparel)

# Plot the actual and predicted label of a misclassified apparel.
plt.figure(figsize=(8, 6))
plt.imshow(failed_sample, cmap="gray")
plt.title(f"The failed predicted apparel is {predicted_failed_apparel_label} \nwhereas the actual apparel is {actual_failed_apparel_label}")
plt.axis('off')
plt.show()
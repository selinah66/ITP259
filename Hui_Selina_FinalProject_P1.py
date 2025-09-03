# Selina Hui
# ITP 259 Fall 2024
# Final Project
# Problem 1

# Chinese Numbers MNIST CNN Model
import pandas as pd
import seaborn as sb
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization

# Load Chinese MNIST data and make DataFrame
chineseMNIST = pd.read_csv("/Users/Selina/Documents/ITP259/FinalProject/chineseMNIST.csv")
pd.set_option("display.max_columns", None)
chinese_numbers = pd.DataFrame(chineseMNIST)

# Define Feature variables and
X = chinese_numbers.iloc[:,:-2] # X is Arabic Number
y = chinese_numbers.iloc[:,-2] # y is Chinese Labels

print(X.shape)
print(y.shape)

# Question 1: Plot the Count of each Chinese number
plt.figure(1, figsize=(10, 10))

ax = sb.countplot(x="label", data=chinese_numbers)

plt.title('Histogram of Chinese Numbers')
plt.xlabel('Number')
plt.ylabel('Frequency')
plt.show()

# Question 2: Visualize 25 random Chinese numbers from the dataset
random_index = np.random.choice(len(X), 25, replace=False)

# Selects row/labels and converts to NumPy array
X_samples = X.iloc[random_index].to_numpy()
y_labels = chinese_numbers.iloc[:,-2][random_index].to_numpy()
chinese_labels = chinese_numbers.iloc[:,-1][random_index].to_numpy()

# Configure Chinese font properties
font_path = '/Users/Selina/Documents/ITP259/FinalProject/SimHei.ttf'  # Replace with your font path
chinese_font = FontProperties(fname=font_path)

# Plot character visualization
plt.figure(1, figsize=(12, 12))
fig, axs = plt.subplots(5, 5, figsize=(10, 10))

for i, ax in enumerate(axs.flat):
  img = X_samples[i].reshape(64, 64)
  ax.set_title(chinese_labels[i], fontproperties=chinese_font, fontsize=12)
  ax.set_xlabel(f'{y_labels[i]}', fontsize=12, labelpad=2)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.imshow(img, cmap='gray')

plt.suptitle("25 Random Chinese Numbers")
plt.tight_layout()
plt.show()

# Questino 3: Scale the pixel values
X = X.astype('float32') / 255.0

# Map dictionary for Chinese Number to its Arabic Number Label
number_to_label = {
    0: 0,     # 0
    1: 1,     # 1
    2: 2,     # 2
    3: 3,     # 3
    4: 4,     # 4
    5: 5,     # 5
    6: 6,     # 6
    7: 7,     # 7
    8: 8,     # 8
    9: 9,     # 9
    10: 10,   # 十 (10)
    100: 11,  # 百 (100)
    1000: 12, # 千 (1000)
    10000: 13,# 万 (10000)
    100000000: 14 # 亿 (100000000)
}

# Map dictionary for Chinese Number to its Chinese Character Label
number_to_chinese_label = {
    0: '零',
    1: '一',
    2: '二',
    3: '三',
    4: '四',
    5: '五',
    6: '六',
    7: '七',
    8: '八',
    9: '九',
    10: '十',
    11: '百',
    12: '千',
    13: '万',
    14: '亿'
}

y = pd.Series(y).map(number_to_label).values

# Reshape X for Model
X = X.to_numpy().reshape(-1, 64, 64, 1)

# Question 4: Partition the dataset into train and test sets & Print the Shapes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2023)
print(f"\nShapes: \n", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Question 5: Build a NN model using Keras layers
model = Sequential([
  # Input layer
  Input(shape=(64, 64, 1)),

  # Convolutional layer 1
  Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
         padding='same', activation='relu'),
  BatchNormalization(),
  MaxPool2D(pool_size=(2, 2)),
  Dropout(0.25),

  # Convolutional Layer 2
  Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
         padding='same', activation='relu'),
  BatchNormalization(),
  MaxPool2D(pool_size=(2, 2)),
  Dropout(0.25),

  # Flatten
  Flatten(),

  # Hidden Dense layers
  Dense(128, activation='relu'),
  Dropout(0.5),
  Dense(128, activation='relu'),
  Dropout(0.5),

  # Output layer
  Dense(15, activation='softmax')
])

# Question 6: Display the model summary
model.summary()

# Question 7: Compile the Sequential model with loss function
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Question 8: Train the model for at least 25 epochs
history = model.fit(X_train, y_train, batch_size=64, epochs=30, validation_data=(X_test, y_test))

# Question 9: Plot accuracy and loss curves
# Loss curve
plt.figure(figsize=[6,6])
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)

# Accuracy curve
plt.figure(figsize=[6,6])
plt.plot(history.history['accuracy'], 'black', linewidth=2.0)
plt.plot(history.history['val_accuracy'], 'blue', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)

# Predict using CNN
pred = model.predict(X_test)

# Convert the predictions into labels
pred_classes = np.argmax(pred, axis=1)
print(pred_classes.shape)

y_test = y_test.astype(int)
pred_classes = pred_classes.astype(int)

# Question 10: Visualize first 30 predicted and actual Chinese character image labels
plt.figure(figsize=[12,10])
plt.suptitle('First 30 Numbers Predicted & Actual Labels', fontsize=15)
for i in range(30):
    plt.subplot(6, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i].reshape(64, 64), cmap='gray')

    true_label = number_to_chinese_label[int(y_test[i])]
    pred_label = number_to_chinese_label[int(pred_classes[i])]

    plt.title(f"Predict: {pred_label}\nTrue: {true_label}",
              fontproperties=chinese_font, fontsize=12)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

# Question 11: Visualize 30 random misclassified images
pred_classes = np.argmax(pred, axis=1)
true_classes = y_test
misclassified = np.where(pred_classes != true_classes)[0][:30]

plt.figure(figsize=[12,10])
plt.suptitle('30 Random Misclassified Chinese Numbers', fontsize=15)
for i, idx in enumerate(misclassified):
    plt.subplot(5, 6, i + 1)
    plt.imshow(X_test[idx].reshape(64, 64), cmap='gray')
    plt.axis('off')

    true_label = number_to_chinese_label[int(true_classes[idx])]
    misclassified_label = number_to_chinese_label[int(pred_classes[idx])]

    plt.title(f'Predicted:{misclassified_label}\nTrue:{true_label}',
              fontproperties=chinese_font, fontsize=12)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
from tensorflow.keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Question 1 + 2: Load Cifar100 Data and partition the dataset into train and test.
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode="fine")

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Dataset Labels
class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
               'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
               'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
               'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
               'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
               'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
               'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon',
               'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
               'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
               'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
               'willow_tree', 'wolf', 'woman', 'worm']

# Question 3: Visualize 30 images from the train dataset
plt.figure(figsize=(12, 10))
plt.suptitle('First 30 Images from Training Dataset', fontsize=14)
for i in range (30):
    plt.subplot(5, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(class_names[y_train[i][0]])
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

# Question 4: Scale the Pixel Values
X_train = X_train/255.0
X_test = X_test/255.0

# Question 5: One-hot Encoding for Classes
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = to_categorical(y_train, 100)
Y_test = to_categorical(y_test, 100)
print("Shape after one-hot encoding: ", Y_train.shape)

# Question 6: Create CNN model
model = Sequential()

# First Convolutional Layer
model.add(Input(shape=(32, 32, 3)))
model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))

# Second Convolutional Layer
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))

# Flatten layer
model.add(Flatten())

# Hidden Dense Layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(100, activation='softmax'))

# Question 7: Compile the model
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Question 8: Train and fit the model
history = model.fit(X_train, Y_train, batch_size=64, epochs=30, verbose=1, validation_data=(X_test, Y_test))

# Question 9: Plot the Loss and Accuracy Curves
plt.figure(figsize=[10,6])
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'green', linewidth=2.0)

plt.legend(['Training Loss','Validation Loss'], fontsize=12)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss Curves',fontsize=14)
plt.show()

# Accuracy Curve
plt.figure(figsize=[10,6])
plt.plot(history.history['accuracy'], 'red', linewidth=2.0)
plt.plot(history.history['val_accuracy'], 'blue', linewidth=2.0)

plt.legend(['Training Accuracy','Validation Accuracy'], fontsize=12)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy Curves',fontsize=14)
plt.show()

# Question 10: Visualize predicted and actual image labels for first 30 images
pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)

# Plot the Actual vs. Predicted results
plt.figure(figsize=[8,10])
plt.suptitle('First 30 Images Predicted & Actual Labels', fontsize=14)
for i in range(30):
    plt.subplot(6, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i])
    plt.title("True: %s\nPredict: %s" %
              (class_names[y_test[i][0]], class_names[pred_classes[i]]), # label in y-test mapped to real classes
              fontsize=8)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

# Question 11: Visualize 30 random misclassified images
# Indices of Misclassified Images
pred_classes = np.argmax(pred, axis=1)
true_classes = np.argmax(Y_test, axis=1)
misclassified = np.where(pred_classes != true_classes)[0][:30]

plt.figure(figsize=[15,15])
plt.suptitle('30 Random Misclassified Images', fontsize=15)
for i, idx in enumerate(misclassified):
    plt.subplot(5, 6, i + 1)
    plt.imshow(X_test[i])
    plt.axis('off')
    plt.title(f'True: {class_names[true_classes[idx]]}\nPredict: {class_names[pred_classes[idx]]}',
              color='red', fontsize=8)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# Define the directory containing the image data
data_dir = "D:/02 Win sem 22 23/LSM/project trial 1/TRAIN"
# Define the target image size
img_size = (128, 128)
# Define the labels for the two classes
labels = ["dry", "wet"]
# Load the image data and labels
data = []
label = []
for i in labels:
    path = os.path.join(data_dir, i)
    class_num = labels.index(i)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            img_resized = cv2.resize(img_array, img_size)
            data.append(img_resized)
            label.append(class_num)
        except Exception as e:
            pass
# Convert the data and labels to NumPy arrays
data = np.array(data)
label = np.array(label)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)
# Save the model
model.save("keras_model.h5")

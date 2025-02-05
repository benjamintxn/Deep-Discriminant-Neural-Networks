# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# **1️⃣ Load and Preprocess the MNIST Dataset**
(x_train, labels_train), (x_test, labels_test) = mnist.load_data()

# Normalize pixel values to [0,1] range
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert labels to one-hot encoding
y_train = to_categorical(labels_train, 10)
y_test = to_categorical(labels_test, 10)

# Reshape data for CNN input (batch_size, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# **2️⃣ Define the CNN Model**
net = Sequential([
    Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)),
    MaxPool2D(pool_size=(2,2)),

    Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2,2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(rate=0.5),  # Dropout to reduce overfitting
    Dense(10, activation='softmax')  # Output layer (10 classes)
])

# **3️⃣ Compile the Model**
net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# **4️⃣ Train the Model**
history = net.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=256)

# **5️⃣ Save the Trained Model**
net.save("mnist_classifier.h5")

# **6️⃣ Reload the Model (Optional - to verify it saves correctly)**
net = load_model("mnist_classifier.h5")

# **7️⃣ Evaluate Performance on Test Data**
outputs = net.predict(x_test)
labels_predicted = np.argmax(outputs, axis=1)
misclassified = sum(labels_predicted != labels_test)
accuracy = 100 * (1 - misclassified / labels_test.size)
print(f'Final Test Accuracy: {accuracy:.2f}%')

# **8️⃣ Plot Training Loss and Validation Loss**
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# **9️⃣ Visualize Sample Predictions**
plt.figure(figsize=(8, 2))
for i in range(0, 8):
    ax = plt.subplot(2, 8, i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray_r')
    plt.title(labels_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

for i in range(0, 8):
    output = net.predict(x_test[i].reshape(1, 28,28,1))
    output = output[0, 0:]
    plt.subplot(2, 8, 8+i+1)
    plt.bar(range(10), output)
    plt.title(np.argmax(output))

plt.show()
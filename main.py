# Import necessary libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import torchvision
from torchvision import transforms
from sklearn.utils import shuffle

# **1️⃣ Function to Load Local Dataset from .h5 File**
def load_data(filepath='all_written_datasets.h5', dataset_name='mnist', reshape_to_cnn=True):
    with h5py.File(filepath, 'r') as f:
        group = f[dataset_name]
        images = group['images'][:]
        labels = group['labels'][:]

        if reshape_to_cnn:
            images = images.reshape(-1, 28, 28, 1)

        return images, labels

# **2️⃣ Load Local Dataset**
dataset_names = ['mnist', 'emnist', 'usps', 'sklearn_digits']  # Update with the correct dataset names inside your .h5 file
x_local, y_local = [], []

for name in dataset_names:
    x_data, y_data = load_data(dataset_name=name, reshape_to_cnn=True)
    x_local.append(x_data)
    y_local.append(y_data)

x_local = np.concatenate(x_local, axis=0)
y_local = np.concatenate(y_local, axis=0)

# **3️⃣ Load and Preprocess MNIST, EMNIST, and USPS Datasets**

# Load MNIST
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

# Load EMNIST from Torchvision
emnist_transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
emnist_train = torchvision.datasets.EMNIST(root="./data", split='digits', train=True, transform=emnist_transform, download=True)
emnist_test = torchvision.datasets.EMNIST(root="./data", split='digits', train=False, transform=emnist_transform, download=True)

# Convert EMNIST to NumPy arrays
x_train_emnist = np.array([np.array(img[0]) for img in emnist_train]) * 255  # Convert to uint8 scale
y_train_emnist = np.array([img[1] for img in emnist_train])

x_test_emnist = np.array([np.array(img[0]) for img in emnist_test]) * 255
y_test_emnist = np.array([img[1] for img in emnist_test])

# Load USPS dataset from torchvision
usps_transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
usps_train = torchvision.datasets.USPS(root="./data", train=True, transform=usps_transform, download=True)
usps_test = torchvision.datasets.USPS(root="./data", train=False, transform=usps_transform, download=True)

# Convert USPS to NumPy arrays
x_train_usps = np.array([np.array(img[0]) for img in usps_train]) * 255  # Convert to uint8 scale
y_train_usps = np.array([img[1] for img in usps_train])

x_test_usps = np.array([np.array(img[0]) for img in usps_test]) * 255
y_test_usps = np.array([img[1] for img in usps_test])

# **4️⃣ Ensure All Datasets Have the Same Shape**
# Reshape to 4D (num_samples, 28, 28, 1)
x_train_mnist = x_train_mnist.reshape(-1, 28, 28, 1)
x_test_mnist = x_test_mnist.reshape(-1, 28, 28, 1)

x_train_emnist = x_train_emnist.reshape(-1, 28, 28, 1)
x_test_emnist = x_test_emnist.reshape(-1, 28, 28, 1)

x_train_usps = x_train_usps.reshape(-1, 28, 28, 1)
x_test_usps = x_test_usps.reshape(-1, 28, 28, 1)

# Normalize pixel values to [0,1]
x_train_mnist = x_train_mnist.astype('float32') / 255
x_test_mnist = x_test_mnist.astype('float32') / 255

x_train_emnist = x_train_emnist.astype('float32') / 255
x_test_emnist = x_test_emnist.astype('float32') / 255

x_train_usps = x_train_usps.astype('float32') / 255
x_test_usps = x_test_usps.astype('float32') / 255

x_local = x_local.astype('float32') / 255

# **5️⃣ Merge All Datasets**
# Print dataset sizes before merging
print(f"MNIST Training: {x_train_mnist.shape[0]} samples")
print(f"EMNIST Training: {x_train_emnist.shape[0]} samples")
print(f"USPS Training: {x_train_usps.shape[0]} samples")
print(f"Local HDF5 Dataset Training: {x_local.shape[0]} samples")

# Merge datasets
x_train = np.concatenate((x_train_mnist, x_train_emnist, x_train_usps, x_local), axis=0)
y_train = np.concatenate((y_train_mnist, y_train_emnist, y_train_usps, y_local), axis=0)

# Print final dataset size after merging
print(f"Total Training Samples: {x_train.shape[0]}")
print(f"Total Training Labels: {y_train.shape[0]}")

# Shuffle dataset to mix all sources properly
x_train, y_train = shuffle(x_train, y_train, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)

# **6️⃣ Data Augmentation for Better Generalization**
datagen = ImageDataGenerator(
    rotation_range=15,  
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    zoom_range=0.1,  
    shear_range=10,  
    horizontal_flip=False,  
    fill_mode='nearest'  
)

augmented_images, augmented_labels = [], []

for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=256, shuffle=False):
    augmented_images.append(x_batch)
    augmented_labels.append(y_batch)
    
    # Stop after generating enough new data (e.g., augmenting all images once)
    if len(augmented_images) * 256 >= len(x_train):
        break

# Convert lists to NumPy arrays
x_augmented = np.concatenate(augmented_images, axis=0)
y_augmented = np.concatenate(augmented_labels, axis=0)

# Merge with original dataset
x_train_full = np.concatenate((x_train, x_augmented), axis=0)
y_train_full = np.concatenate((y_train, y_augmented), axis=0)

print(f"New Training Set Size After Augmentation: {x_train_full.shape[0]}")

datagen.fit(x_train)

# **6️⃣ Define the CNN Model BEFORE Training**
net = Sequential([
    Conv2D(filters=64, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),

    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),

    Conv2D(filters=64, kernel_size=(3,3), activation='relu'), 
    BatchNormalization(),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(rate=0.5),  
    Dense(256, activation='relu'),
    Dropout(rate=0.3),  
    Dense(10, activation='softmax')
])

# **7️⃣ Compile the Model**
net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# **8️⃣ Print Total Training Samples & Confirm Training**
print(f"Training on {x_train.shape[0]} images with batch size {256}")

# **9️⃣ Train the Model with Augmented Data**
history = net.fit(datagen.flow(x_train_full, y_train_full, batch_size=256), 
                  epochs=30,  
                  steps_per_epoch=len(x_train_full) // 256)

# **🔟 Save the Trained Model**
net.save("multidataset_classifier_augmented.h5")

# **🔟 Reload the Model (Optional - to verify it saves correctly)**
net = load_model("multidataset_classifier_augmented.h5")

# **🔟 Evaluate Performance on Test Data**
outputs = net.predict(x_test_mnist)
labels_predicted = np.argmax(outputs, axis=1)
misclassified = sum(labels_predicted != y_test_mnist)
accuracy = 100 * (1 - misclassified / y_test_mnist.shape[0])
print(f'Final Test Accuracy: {accuracy:.2f}%')

# Test on MNIST
mnist_acc = net.evaluate(x_test_mnist, y_test_mnist, verbose=0)[1]
print(f"Accuracy on MNIST Test Set: {mnist_acc*100:.2f}%")

# Test on EMNIST
emnist_acc = net.evaluate(x_test_emnist, y_test_emnist, verbose=0)[1]
print(f"Accuracy on EMNIST Test Set: {emnist_acc*100:.2f}%")

# Test on USPS
usps_acc = net.evaluate(x_test_usps, y_test_usps, verbose=0)[1]
print(f"Accuracy on USPS Test Set: {usps_acc*100:.2f}%")

# Test on Local HDF5 Dataset
local_acc = net.evaluate(x_local, y_local, verbose=0)[1]
print(f"Accuracy on Local Dataset: {local_acc*100:.2f}%")

# **🔟 Plot Training Loss and Validation Loss**
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

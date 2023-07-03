import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

# Step 1: Preprocess a subset of the ImageNet dataset
data_dir = "/path/to/dataset"  
image_size = (224, 224)
num_classes = 10  

# Load and preprocess images and labels
images = []
labels = []

class_names = os.listdir(data_dir)
class_dict = {class_name: i for i, class_name in enumerate(class_names)}

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image = load_img(image_path, target_size=image_size)
        image = img_to_array(image)
        images.append(image)
        labels.append(class_dict[class_name])

# Convert to numpy arrays
images = np.array(images, dtype="float32") / 255.0
labels = tf.keras.utils.to_categorical(labels, num_classes)

# Split the dataset into training, validation, and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# Step 2: Implement an image classification system
# Load the pre-trained EfficientNetB0 model without the top classification layer
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a new classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 3: Train the image classification system with data augmentation
# Define batch size and number of epochs
batch_size = 32
epochs = 10

# Implement a learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch > 0:
        lr = lr * 0.1
    return lr

lr_callback = LearningRateScheduler(lr_scheduler)

# Train the model
model.fit(
    train_images, train_labels,
    batch_size=batch_size, epochs=epochs,
    validation_data=(val_images, val_labels),
    callbacks=[lr_callback]
)

# Step 4: Evaluate system performance
# Evaluate on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# Step 5: Iterate on the system to improve performance


model.save('image_classification_model.h5')

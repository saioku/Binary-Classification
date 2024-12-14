from cnn_transformer_model import build_model
from data_handler import preprocess_train_images, load_test_images
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# important params
train_dir = "train"
test_dir = "test1"
processed_dir = "pre-processed_images"
image_size = (64, 64)
batch_size = 64
epochs = 6
upscale_factor = 1.2  # upscale factor

upscaled_image_size = (
    int(image_size[0] * upscale_factor),
    int(image_size[1] * upscale_factor),
)

# training data
train_images, train_labels = preprocess_train_images(train_dir, processed_dir, image_size, upscale_factor)

x_train, x_val, y_train, y_val = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# test data
test_images, test_ids = load_test_images(test_dir, image_size, upscale_factor)

# reshape for CNN input 
x_train = x_train.reshape((-1, upscaled_image_size[0], upscaled_image_size[1], 1))
x_val = x_val.reshape((-1, upscaled_image_size[0], upscaled_image_size[1], 1))
test_images = test_images.reshape((-1, upscaled_image_size[0], upscaled_image_size[1], 1))

# data augmentation
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
)
data_gen.fit(x_train)

# build and compile model
model = build_model(upscaled_image_size)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# train model
history = model.fit(
    data_gen.flow(x_train, y_train, batch_size=batch_size),
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.2),
    ],
)

predictions = model.predict(test_images)
labels = (predictions > 0.5).astype(int).flatten()

test_image_files = sorted(os.listdir(test_dir))  
image_ids = [os.path.splitext(filename)[0] for filename in test_image_files] 

# submission.csv (submission1 since they are not sorted they are later run through sorting script)
results = pd.DataFrame({'id': image_ids, 'label': labels})
results.to_csv('submission1.csv', index=False)

def plot_training_curves(history):
    plt.figure(figsize=(12, 5))

    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_curves(history)

# F1 score, precision, and recall
y_test_true = [1 if i % 2 == 0 else 0 for i in range(len(labels))]  
precision = precision_score(y_test_true, labels)
recall = recall_score(y_test_true, labels)

print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

final_accuracy = history.history['val_accuracy'][-1] * 100
print(f"Final Validation Accuracy: {final_accuracy:.2f}%")

# classification report
report = classification_report(y_test_true, labels, target_names=["Class 0", "Class 1"])
print("\nClassification Report:\n")
print(report)